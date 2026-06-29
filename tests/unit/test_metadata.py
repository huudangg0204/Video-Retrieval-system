"""
Unit tests for pipeline/metadata.py.

Tests cover:
1. `KeyframeMetadata` dataclass — round-trip JSON, schema fields.
2. `MetadataExtractor.process_keyframe()` — combines detection + scene
   + scene_graph into a single JSON sidecar, with mocked model inputs.
3. `MetadataExtractor.process_video()` — walks a video folder, skips
   existing sidecars (idempotent re-run), aggregates stats.
4. `MetadataExtractor.process_all()` — walks the keyframes root, mirror
   structure of `data/detections/`.
5. `metadata-stats` aggregator — counts + scene distribution.

The real models are NOT loaded; detector / classifier / scene_graph
extractor are injected as callables returning deterministic dicts /
tuples. This keeps tests fast and deterministic.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from aic26.pipeline.metadata import (
    KeyframeMetadata,
    MetadataExtractor,
    MetadataStats,
    aggregate_metadata_stats,
)

# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """A 320×180 test JPEG."""
    img_path = tmp_path / "frame_0001.jpg"
    Image.new("RGB", (320, 180), color=(200, 100, 50)).save(img_path)
    return img_path


@pytest.fixture
def video_keyframes(tmp_path: Path) -> tuple[Path, Path]:
    """Create one video's keyframe folder with 3 dummy JPEGs.

    Returns (keyframes_root, video_dir).
    """
    kf_root = tmp_path / "keyframes"
    video_dir = kf_root / "L01_V001"
    video_dir.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (320, 180), color=(i * 60, 100, 50)).save(
            video_dir / f"frame_{i:04d}.jpg",
        )
    return kf_root, video_dir


def _stub_detector(image_path: Path) -> dict:
    """Returns a minimal YOLO-style detection dict with positions."""
    return {
        "detection_scores": [0.9, 0.7],
        "detection_class_names": ["/m/01g317", "/m/0dt3t"],  # person, cup
        "detection_class_entities": ["Person", "Cup"],
        "detection_boxes": [[10.0, 10.0, 150.0, 150.0], [200.0, 80.0, 280.0, 160.0]],
        "detection_class_labels": [0, 41],
        "detection_positions": ["top-left", "right"],
    }


def _stub_scene(image: Image.Image) -> dict:
    """Returns a minimal Places365-style dict."""
    return {
        "category": "kitchen",
        "confidence": 0.85,
        "top5": [
            ("kitchen", 0.85),
            ("dining_room", 0.07),
            ("restaurant", 0.04),
            ("living_room", 0.02),
            ("pantry", 0.01),
        ],
    }


def _stub_scene_graph(image: Image.Image) -> list[dict]:
    """Returns a list of scene-graph triple dicts."""
    return [
        {"s": "person", "p": "hold", "o": "cup"},
        {"s": "cup", "p": "on", "o": "table"},
    ]


# ── KeyframeMetadata dataclass ─────────────────────────────────────────


class TestKeyframeMetadata:
    def test_construction(self):
        m = KeyframeMetadata(
            video="L01_V001",
            frame_idx=42,
            detections=[
                {"label": "Person", "score": 0.9, "bbox": [0, 0, 1, 1], "position": "top-left"}
            ],
            scene={"category": "kitchen", "confidence": 0.85, "top5": []},
            scene_graph=[{"s": "person", "p": "hold", "o": "cup"}],
        )
        assert m.video == "L01_V001"
        assert m.frame_idx == 42
        assert len(m.detections) == 1
        assert m.scene["category"] == "kitchen"
        assert len(m.scene_graph) == 1

    def test_to_dict_matches_schema(self):
        m = KeyframeMetadata(
            video="L01_V001",
            frame_idx=42,
            detections=[],
            scene={"category": "kitchen", "confidence": 0.85, "top5": []},
            scene_graph=[],
        )
        d = m.to_dict()
        assert d["video"] == "L01_V001"
        assert d["frame_idx"] == 42
        assert d["detections"] == []
        assert d["scene"]["category"] == "kitchen"
        assert d["scene_graph"] == []

    def test_round_trip_json(self):
        m = KeyframeMetadata(
            video="L01_V001",
            frame_idx=42,
            detections=[
                {"label": "Person", "score": 0.9, "bbox": [0, 0, 1, 1], "position": "top-left"},
            ],
            scene={"category": "kitchen", "confidence": 0.85, "top5": [["kitchen", 0.85]]},
            scene_graph=[{"s": "person", "p": "hold", "o": "cup"}],
        )
        s = m.to_json()
        loaded = KeyframeMetadata.from_json(s)
        assert loaded.video == m.video
        assert loaded.frame_idx == m.frame_idx
        assert loaded.detections == m.detections
        # JSON tuples come back as lists — that's expected for top5.
        assert loaded.scene["category"] == m.scene["category"]
        assert loaded.scene["top5"] == [["kitchen", 0.85]]
        assert loaded.scene_graph == m.scene_graph

    def test_from_dict(self):
        raw = {
            "video": "L01_V001",
            "frame_idx": 7,
            "detections": [
                {"label": "Cup", "score": 0.8, "bbox": [1, 2, 3, 4], "position": "center"}
            ],
            "scene": {"category": "beach", "confidence": 0.7, "top5": []},
            "scene_graph": [],
        }
        m = KeyframeMetadata.from_dict(raw)
        assert m.video == "L01_V001"
        assert m.frame_idx == 7
        assert m.detections[0]["label"] == "Cup"
        assert m.scene["category"] == "beach"

    def test_save_and_load(self, tmp_path: Path):
        m = KeyframeMetadata(
            video="L01_V001",
            frame_idx=42,
            detections=[],
            scene={"category": "kitchen", "confidence": 0.85, "top5": []},
            scene_graph=[],
        )
        out_path = tmp_path / "42.json"
        m.save(out_path)
        assert out_path.exists()
        loaded = KeyframeMetadata.load(out_path)
        assert loaded.frame_idx == 42
        assert loaded.video == "L01_V001"

    def test_detections_optional(self):
        # Should accept None to mean "not run" and persist as empty list.
        m = KeyframeMetadata(
            video="L01_V001",
            frame_idx=0,
            detections=None,  # type: ignore[arg-type]
            scene=None,  # type: ignore[arg-type]
            scene_graph=None,  # type: ignore[arg-type]
        )
        d = m.to_dict()
        assert d["detections"] == []
        assert d["scene"] is None
        assert d["scene_graph"] is None


# ── MetadataExtractor.process_keyframe ────────────────────────────────


class TestProcessKeyframe:
    def _make_extractor(
        self,
        detector: Callable | None = None,
        scene: Callable | None = None,
        scene_graph: Callable | None = None,
    ) -> MetadataExtractor:
        return MetadataExtractor(
            detector_fn=detector or _stub_detector,
            scene_fn=scene or _stub_scene,
            scene_graph_fn=scene_graph or _stub_scene_graph,
        )

    def test_returns_keyframe_metadata(self, sample_image: Path):
        ext = self._make_extractor()
        meta = ext.process_keyframe(sample_image, video="L01_V001", frame_idx=1)
        assert isinstance(meta, KeyframeMetadata)
        assert meta.video == "L01_V001"
        assert meta.frame_idx == 1

    def test_combines_all_three_sources(self, sample_image: Path):
        ext = self._make_extractor()
        meta = ext.process_keyframe(sample_image, video="L01_V001", frame_idx=1)
        assert len(meta.detections) == 2
        assert meta.detections[0]["label"] == "Person"
        assert meta.detections[1]["position"] == "right"
        assert meta.scene["category"] == "kitchen"
        assert len(meta.scene_graph) == 2
        assert meta.scene_graph[0] == {"s": "person", "p": "hold", "o": "cup"}

    def test_skip_scene_graph(self, sample_image: Path):
        ext = self._make_extractor()
        meta = ext.process_keyframe(
            sample_image,
            video="L01_V001",
            frame_idx=1,
            skip_scene_graph=True,
        )
        assert meta.scene_graph is None

    def test_skip_scene(self, sample_image: Path):
        ext = self._make_extractor()
        meta = ext.process_keyframe(
            sample_image,
            video="L01_V001",
            frame_idx=1,
            skip_scene=True,
        )
        assert meta.scene is None

    def test_skip_detections(self, sample_image: Path):
        ext = self._make_extractor()
        meta = ext.process_keyframe(
            sample_image,
            video="L01_V001",
            frame_idx=1,
            skip_detections=True,
        )
        assert meta.detections == []

    def test_detector_returns_empty_dict(self, sample_image: Path):
        # If detector yields no detections, detections list should be empty,
        # not None — downstream consumers expect a list.
        ext = self._make_extractor(detector=lambda p: {})
        meta = ext.process_keyframe(sample_image, video="L01_V001", frame_idx=1)
        assert meta.detections == []

    def test_scene_graph_extractor_returns_empty(self, sample_image: Path):
        ext = self._make_extractor(scene_graph=lambda img: [])
        meta = ext.process_keyframe(sample_image, video="L01_V001", frame_idx=1)
        assert meta.scene_graph == []

    def test_handles_corrupt_detector_output(self, sample_image: Path):
        # Detector returns None — extractor should treat as empty.
        ext = self._make_extractor(detector=lambda p: None)
        meta = ext.process_keyframe(sample_image, video="L01_V001", frame_idx=1)
        assert meta.detections == []

    def test_handles_corrupt_image(self, tmp_path: Path):
        bad = tmp_path / "broken.jpg"
        bad.write_bytes(b"not an image")
        ext = self._make_extractor()
        # Unreadable image → process_keyframe returns None so caller
        # can mark it as failed and skip writing a sidecar.
        meta = ext.process_keyframe(bad, video="L01_V001", frame_idx=1)
        assert meta is None

    def test_frame_idx_from_filename(self, sample_image: Path):
        ext = self._make_extractor()
        meta = ext.process_keyframe(sample_image, video="L01_V001")
        # Default extraction reads frame_idx from filename stem
        assert meta.frame_idx == 1  # "frame_0001.jpg" → 1

    def test_video_from_path(self, sample_image: Path):
        ext = self._make_extractor()
        meta = ext.process_keyframe(sample_image, video="L01_V001")
        assert meta.video == "L01_V001"


# ── MetadataExtractor.process_video (idempotent walk) ──────────────────


class TestProcessVideo:
    def _make_extractor(self) -> MetadataExtractor:
        return MetadataExtractor(
            detector_fn=_stub_detector,
            scene_fn=_stub_scene,
            scene_graph_fn=_stub_scene_graph,
        )

    def test_walks_video_folder(self, video_keyframes: tuple[Path, Path], tmp_path: Path):
        kf_root, video_dir = video_keyframes
        output_root = tmp_path / "metadata"
        ext = self._make_extractor()
        stats = ext.process_video(video_dir=video_dir, output_root=output_root)
        assert stats["frames_total"] == 3
        assert stats["frames_processed"] == 3
        assert stats["frames_failed"] == 0
        assert (output_root / "L01_V001").exists()
        # 3 sidecars written
        sidecars = list((output_root / "L01_V001").glob("*.json"))
        assert len(sidecars) == 3

    def test_output_filename_matches_frame_stem(self, video_keyframes, tmp_path):
        kf_root, video_dir = video_keyframes
        output_root = tmp_path / "metadata"
        ext = self._make_extractor()
        ext.process_video(video_dir=video_dir, output_root=output_root)
        names = sorted(p.name for p in (output_root / "L01_V001").glob("*.json"))
        assert names == ["frame_0000.json", "frame_0001.json", "frame_0002.json"]

    def test_sidecar_content(self, video_keyframes, tmp_path):
        kf_root, video_dir = video_keyframes
        output_root = tmp_path / "metadata"
        ext = self._make_extractor()
        ext.process_video(video_dir=video_dir, output_root=output_root)
        sidecar = json.loads((output_root / "L01_V001" / "frame_0000.json").read_text())
        assert sidecar["video"] == "L01_V001"
        assert sidecar["frame_idx"] == 0
        assert "scene" in sidecar
        assert "scene_graph" in sidecar

    def test_idempotent_skip_existing(self, video_keyframes, tmp_path):
        # First run creates all 3 sidecars. Second run should skip them.
        kf_root, video_dir = video_keyframes
        output_root = tmp_path / "metadata"
        ext = self._make_extractor()
        first = ext.process_video(video_dir=video_dir, output_root=output_root)
        second = ext.process_video(video_dir=video_dir, output_root=output_root)
        assert first["frames_processed"] == 3
        assert second["frames_processed"] == 0
        assert second["frames_skipped"] == 3

    def test_overwrite_existing_when_forced(self, video_keyframes, tmp_path):
        kf_root, video_dir = video_keyframes
        output_root = tmp_path / "metadata"
        ext = self._make_extractor()
        ext.process_video(video_dir=video_dir, output_root=output_root)
        third = ext.process_video(
            video_dir=video_dir,
            output_root=output_root,
            overwrite=True,
        )
        assert third["frames_processed"] == 3

    def test_partial_corruption_does_not_halt(self, video_keyframes, tmp_path):
        kf_root, video_dir = video_keyframes
        # Corrupt one of the 3 keyframes
        (video_dir / "frame_0001.jpg").write_bytes(b"not jpeg")
        output_root = tmp_path / "metadata"
        ext = self._make_extractor()
        stats = ext.process_video(video_dir=video_dir, output_root=output_root)
        assert stats["frames_total"] == 3
        assert stats["frames_processed"] == 2
        assert stats["frames_failed"] == 1

    def test_only_processes_image_files(self, video_keyframes, tmp_path):
        kf_root, video_dir = video_keyframes
        # Drop a non-image file alongside the JPEGs
        (video_dir / "notes.txt").write_text("hello")
        output_root = tmp_path / "metadata"
        ext = self._make_extractor()
        stats = ext.process_video(video_dir=video_dir, output_root=output_root)
        assert stats["frames_total"] == 3  # .txt ignored

    def test_passes_skip_flags_through(self, video_keyframes, tmp_path):
        kf_root, video_dir = video_keyframes
        output_root = tmp_path / "metadata"
        ext = self._make_extractor()
        ext.process_video(
            video_dir=video_dir,
            output_root=output_root,
            skip_scene_graph=True,
        )
        sidecar = json.loads((output_root / "L01_V001" / "frame_0000.json").read_text())
        assert sidecar["scene_graph"] is None
        assert sidecar["scene"] is not None  # still ran


# ── MetadataExtractor.process_all (multi-video) ────────────────────────


class TestProcessAll:
    def _make_extractor(self) -> MetadataExtractor:
        return MetadataExtractor(
            detector_fn=_stub_detector,
            scene_fn=_stub_scene,
            scene_graph_fn=_stub_scene_graph,
        )

    def test_walks_all_videos(self, tmp_path):
        kf_root = tmp_path / "keyframes"
        for video in ("L01_V001", "L01_V002"):
            video_dir = kf_root / video
            video_dir.mkdir(parents=True)
            for i in range(2):
                Image.new("RGB", (32, 32), color=(i, i, i)).save(
                    video_dir / f"frame_{i:04d}.jpg",
                )
        output_root = tmp_path / "metadata"
        ext = self._make_extractor()
        stats = ext.process_all(keyframes_root=kf_root, output_root=output_root)
        assert stats["videos_total"] == 2
        assert stats["frames_total"] == 4
        assert (output_root / "L01_V001").exists()
        assert (output_root / "L01_V002").exists()

    def test_empty_root(self, tmp_path):
        kf_root = tmp_path / "keyframes"
        kf_root.mkdir()
        output_root = tmp_path / "metadata"
        ext = self._make_extractor()
        stats = ext.process_all(keyframes_root=kf_root, output_root=output_root)
        assert stats["videos_total"] == 0
        assert stats["frames_total"] == 0

    def test_skips_non_dir_entries(self, tmp_path):
        kf_root = tmp_path / "keyframes"
        kf_root.mkdir()
        (kf_root / "README.txt").write_text("hi")
        video_dir = kf_root / "L01_V001"
        video_dir.mkdir()
        Image.new("RGB", (32, 32)).save(video_dir / "frame_0000.jpg")
        output_root = tmp_path / "metadata"
        ext = self._make_extractor()
        stats = ext.process_all(keyframes_root=kf_root, output_root=output_root)
        assert stats["videos_total"] == 1


# ── aggregate_metadata_stats ──────────────────────────────────────────


class TestAggregateStats:
    def test_aggregates_scene_distribution(self, tmp_path):
        meta_root = tmp_path / "metadata"
        meta_root.mkdir()
        for video in ("L01_V001", "L01_V002"):
            vdir = meta_root / video
            vdir.mkdir()
            # Write 3 sidecars with varying scenes
            scenes = ["kitchen", "kitchen", "beach"]
            for i, scene in enumerate(scenes):
                KeyframeMetadata(
                    video=video,
                    frame_idx=i,
                    detections=[],
                    scene={"category": scene, "confidence": 0.8, "top5": []},
                    scene_graph=[{"s": "person", "p": "sit", "o": "chair"}],
                ).save(vdir / f"frame_{i:04d}.json")
        stats = aggregate_metadata_stats(meta_root)
        assert isinstance(stats, MetadataStats)
        assert stats.total_sidecars == 6
        assert stats.total_videos == 2
        # Scene distribution: 4 kitchen, 2 beach
        assert stats.scene_distribution["kitchen"] == 4
        assert stats.scene_distribution["beach"] == 2
        assert stats.scene_graph_total_triples == 6  # 6 frames × 1 triple

    def test_handles_null_scene(self, tmp_path):
        meta_root = tmp_path / "metadata"
        meta_root.mkdir()
        vdir = meta_root / "L01_V001"
        vdir.mkdir()
        KeyframeMetadata(
            video="L01_V001",
            frame_idx=0,
            detections=[],
            scene=None,
            scene_graph=None,  # type: ignore[arg-type]
        ).save(vdir / "frame_0000.json")
        stats = aggregate_metadata_stats(meta_root)
        assert stats.total_sidecars == 1
        assert stats.scene_distribution == {}
        assert stats.scene_graph_total_triples == 0

    def test_avg_triples_per_keyframe(self, tmp_path):
        meta_root = tmp_path / "metadata"
        meta_root.mkdir()
        vdir = meta_root / "L01_V001"
        vdir.mkdir()
        # First sidecar: 3 triples. Second: 1 triple.
        KeyframeMetadata(
            video="L01_V001",
            frame_idx=0,
            detections=[],
            scene=None,  # type: ignore[arg-type]
            scene_graph=[
                {"s": "a", "p": "b", "o": "c"},
                {"s": "d", "p": "e", "o": "f"},
                {"s": "g", "p": "h", "o": "i"},
            ],
        ).save(vdir / "frame_0000.json")
        KeyframeMetadata(
            video="L01_V001",
            frame_idx=1,
            detections=[],
            scene=None,  # type: ignore[arg-type]
            scene_graph=[{"s": "x", "p": "y", "o": "z"}],
        ).save(vdir / "frame_0001.json")
        stats = aggregate_metadata_stats(meta_root)
        assert stats.scene_graph_total_triples == 4
        assert stats.avg_triples_per_keyframe == 2.0

    def test_empty_root(self, tmp_path):
        meta_root = tmp_path / "metadata"
        meta_root.mkdir()
        stats = aggregate_metadata_stats(meta_root)
        assert stats.total_sidecars == 0
        assert stats.total_videos == 0
        assert stats.scene_distribution == {}


# ── Schema contract ────────────────────────────────────────────────────


class TestSchemaContract:
    """Locks the JSON shape so downstream consumers don't break."""

    def test_required_top_level_keys(self, tmp_path):
        m = KeyframeMetadata(
            video="L01_V001",
            frame_idx=0,
            detections=[],
            scene=None,
            scene_graph=None,  # type: ignore[arg-type]
        )
        d = m.to_dict()
        assert set(d.keys()) >= {
            "video",
            "frame_idx",
            "detections",
            "scene",
            "scene_graph",
        }

    def test_detection_entry_shape(self, tmp_path):
        m = KeyframeMetadata(
            video="L01_V001",
            frame_idx=0,
            detections=[
                {"label": "Person", "score": 0.9, "bbox": [1, 2, 3, 4], "position": "top-left"}
            ],
            scene=None,
            scene_graph=None,  # type: ignore[arg-type]
        )
        d = m.to_dict()
        det = d["detections"][0]
        assert set(det.keys()) >= {"label", "score", "bbox", "position"}

    def test_scene_graph_entry_shape(self, tmp_path):
        m = KeyframeMetadata(
            video="L01_V001",
            frame_idx=0,
            detections=[],
            scene=None,  # type: ignore[arg-type]
            scene_graph=[{"s": "person", "p": "hold", "o": "cup"}],
        )
        d = m.to_dict()
        sg = d["scene_graph"][0]
        assert set(sg.keys()) >= {"s", "p", "o"}
