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
6. `load_precomputed_detections` — backfills `detection_positions` on
   older detection JSONs that lack the field (Task 1.1).
7. Scene confidence thresholding + `scene_group` aggregation (Task 2.2).

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
    load_precomputed_detections,
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

    def test_nonexistent_root(self, tmp_path):
        # Missing dir → returns zero-filled stats, no exception.
        meta_root = tmp_path / "does-not-exist"
        stats = aggregate_metadata_stats(meta_root)
        assert stats.total_sidecars == 0
        assert stats.total_videos == 0

    def test_repr_with_distribution(self):
        s = MetadataStats(
            total_sidecars=10, total_videos=2,
            scene_distribution={"kitchen": 7, "beach": 3},
            scene_graph_total_triples=15, avg_triples_per_keyframe=1.5,
        )
        # Should show top scene in the repr
        r = repr(s)
        assert "kitchen" in r
        assert "10" in r

    def test_repr_empty(self):
        s = MetadataStats()
        r = repr(s)
        assert "n/a" in r

    def test_skips_malformed_sidecar(self, tmp_path):
        meta_root = tmp_path / "metadata"
        vdir = meta_root / "L01_V001"
        vdir.mkdir(parents=True)
        # Write one valid and one corrupt sidecar
        KeyframeMetadata(
            video="L01_V001", frame_idx=0, detections=[],
            scene={"category": "kitchen", "confidence": 0.8, "top5": []},
            scene_graph=[],
        ).save(vdir / "frame_0000.json")
        (vdir / "frame_0001.json").write_text("not json at all")
        stats = aggregate_metadata_stats(meta_root)
        # Both files are counted; corrupt one is skipped during
        # aggregation (no scene, no triples contribute to stats).
        assert stats.total_sidecars == 2
        # The valid sidecar still drives the scene distribution.
        assert stats.scene_distribution.get("kitchen") == 1
        assert stats.scene_graph_total_triples == 0


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


# ── Task 1.1: position enrichment in load_precomputed_detections ──────


def _write_detection(video_dir: Path, stem: str, det: dict) -> Path:
    """Write a detection JSON and return its path."""
    det_path = video_dir / f"{stem}.json"
    det_path.write_text(json.dumps(det), encoding="utf-8")
    return det_path


def _bare_detection(bbox: list[float]) -> dict:
    """Build a detection dict WITHOUT detection_positions (legacy shape)."""
    return {
        "detection_scores": [0.9],
        "detection_class_names": ["/m/01g317"],
        "detection_class_entities": ["Person"],
        "detection_boxes": [bbox],
        "detection_class_labels": [0],
    }


class TestLoadPrecomputedDetections:
    """`load_precomputed_detections` should backfill positions for legacy files."""

    def test_returns_none_when_no_file(self, tmp_path: Path):
        det_dir = tmp_path / "detections"
        det_dir.mkdir()
        (det_dir / "V001").mkdir()
        image_path = tmp_path / "keyframes" / "V001" / "001.jpg"
        (tmp_path / "keyframes" / "V001").mkdir(parents=True)
        Image.new("RGB", (640, 360)).save(image_path)

        fn = load_precomputed_detections(det_dir)
        assert fn(image_path) is None

    def test_reads_legacy_file_without_positions_key(self, tmp_path: Path):
        """A detection file lacking detection_positions is still read."""
        det_dir = tmp_path / "detections"
        (det_dir / "V001").mkdir(parents=True)
        _write_detection(
            det_dir / "V001",
            "001",
            _bare_detection([10, 10, 160, 160]),  # 150×150 = 9.77% of 640×360
        )
        kf_dir = tmp_path / "keyframes" / "V001"
        kf_dir.mkdir(parents=True)
        image_path = kf_dir / "001.jpg"
        Image.new("RGB", (640, 360)).save(image_path)

        fn = load_precomputed_detections(det_dir)
        det = fn(image_path)
        assert det is not None
        assert "detection_positions" in det
        assert det["detection_positions"] == ["top-left"]

    def test_existing_positions_are_not_overwritten(self, tmp_path: Path):
        """A newer detection file with positions must be returned unchanged."""
        det_dir = tmp_path / "detections"
        (det_dir / "V001").mkdir(parents=True)
        det = _bare_detection([10, 10, 160, 160])
        det["detection_positions"] = ["center"]  # intentionally wrong on purpose
        _write_detection(det_dir / "V001", "001", det)

        kf_dir = tmp_path / "keyframes" / "V001"
        kf_dir.mkdir(parents=True)
        image_path = kf_dir / "001.jpg"
        Image.new("RGB", (640, 360)).save(image_path)

        fn = load_precomputed_detections(det_dir)
        assert fn(image_path)["detection_positions"] == ["center"]

    def test_corrupt_image_returns_detection_without_positions(self, tmp_path: Path):
        """If the keyframe can't be opened, fall back to the bare detection."""
        det_dir = tmp_path / "detections"
        (det_dir / "V001").mkdir(parents=True)
        _write_detection(det_dir / "V001", "001", _bare_detection([10, 10, 160, 160]))

        kf_dir = tmp_path / "keyframes" / "V001"
        kf_dir.mkdir(parents=True)
        image_path = kf_dir / "001.jpg"
        # Write a non-image file to force PIL failure.
        image_path.write_text("not an image", encoding="utf-8")

        fn = load_precomputed_detections(det_dir)
        det = fn(image_path)
        assert det is not None
        assert "detection_positions" not in det  # graceful: don't crash, don't enrich

    def test_small_bbox_gets_none_position(self, tmp_path: Path):
        """A bbox below the 5% threshold should map to None."""
        det_dir = tmp_path / "detections"
        (det_dir / "V001").mkdir(parents=True)
        # 10×10 = 100 px = 0.04% of 640×360 → well below 5% threshold.
        _write_detection(det_dir / "V001", "001", _bare_detection([100, 100, 110, 110]))

        kf_dir = tmp_path / "keyframes" / "V001"
        kf_dir.mkdir(parents=True)
        image_path = kf_dir / "001.jpg"
        Image.new("RGB", (640, 360)).save(image_path)

        fn = load_precomputed_detections(det_dir)
        det = fn(image_path)
        assert det["detection_positions"] == [None]

    def test_multiple_bboxes_all_get_positions(self, tmp_path: Path):
        det_dir = tmp_path / "detections"
        (det_dir / "V001").mkdir(parents=True)
        det = {
            "detection_scores": [0.9, 0.8, 0.7],
            "detection_class_names": ["a", "b", "c"],
            "detection_class_entities": ["A", "B", "C"],
            "detection_boxes": [
                [10, 10, 160, 160],   # top-left
                [245, 110, 395, 250],  # center
                [480, 200, 630, 350],  # bottom-right
            ],
            "detection_class_labels": [0, 1, 2],
        }
        _write_detection(det_dir / "V001", "001", det)

        kf_dir = tmp_path / "keyframes" / "V001"
        kf_dir.mkdir(parents=True)
        image_path = kf_dir / "001.jpg"
        Image.new("RGB", (640, 360)).save(image_path)

        fn = load_precomputed_detections(det_dir)
        out = fn(image_path)
        assert out["detection_positions"] == ["top-left", "center", "bottom-right"]

    def test_custom_min_bbox_fraction_respected(self, tmp_path: Path):
        """A larger threshold (10%) should reject bboxes that 5% would accept."""
        det_dir = tmp_path / "detections"
        (det_dir / "V001").mkdir(parents=True)
        # 150×150 = 9.77% → passes 5% but fails 10%.
        _write_detection(det_dir / "V001", "001", _bare_detection([10, 10, 160, 160]))

        kf_dir = tmp_path / "keyframes" / "V001"
        kf_dir.mkdir(parents=True)
        image_path = kf_dir / "001.jpg"
        Image.new("RGB", (640, 360)).save(image_path)

        fn = load_precomputed_detections(det_dir, min_bbox_fraction=0.10)
        assert fn(image_path)["detection_positions"] == [None]


# ── Task 2.2: scene confidence thresholding + scene_group ──────────────


def _make_scene_fn_from_top1(top1_category: str, top1_confidence: float, top5: list):
    """Build a `_scene_fn` matching the production one in build_metadata_callables.

    Mirrors `pipeline/metadata.py::_scene_fn` so we can test the threshold
    + scene_group aggregation logic in isolation, without loading Places365.
    """
    from collections import Counter

    from aic26.pipeline.places365 import _infer_scene_group

    def _scene_fn(image: Image.Image) -> dict:
        category = top1_category
        confidence = top1_confidence

        if confidence < 0.3:
            category = "uncertain"
            top5_groups = [_infer_scene_group(lbl) for lbl, _ in top5]
            group_counter = Counter(top5_groups)
            scene_group = group_counter.most_common(1)[0][0]
        else:
            scene_group = _infer_scene_group(category)

        return {
            "category": category,
            "confidence": confidence,
            "top5": top5,
            "scene_group": scene_group,
        }

    return _scene_fn


class TestSceneConfidenceThresholding:
    """Task 2.2: low-confidence scenes are flagged 'uncertain' with scene_group fallback."""

    def test_high_confidence_returns_category_unchanged(self):
        top5 = [
            ("kitchen", 0.85),
            ("dining_room", 0.07),
            ("restaurant", 0.04),
            ("living_room", 0.02),
            ("pantry", 0.01),
        ]
        fn = _make_scene_fn_from_top1("kitchen", 0.85, top5)
        out = fn(Image.new("RGB", (10, 10)))
        assert out["category"] == "kitchen"
        assert out["scene_group"] == "indoor_residential"  # "kitchen" matches indoor_residential

    def test_low_confidence_marks_uncertain(self):
        top5 = [
            ("pharmacy", 0.25),
            ("general_store_indoor", 0.15),
            ("drugstore", 0.09),
            ("chemistry_lab", 0.08),
            ("biology_laboratory", 0.05),
        ]
        fn = _make_scene_fn_from_top1("pharmacy", 0.25, top5)
        out = fn(Image.new("RGB", (10, 10)))
        assert out["category"] == "uncertain"
        assert out["confidence"] == 0.25  # preserved for transparency

    def test_low_confidence_scene_group_from_top5_majority(self):
        """When uncertain, scene_group is majority-vote of top5 inferred groups."""
        top5 = [
            ("general_store_indoor", 0.25),  # → indoor_commercial
            ("drugstore", 0.20),             # → indoor_commercial
            ("pharmacy", 0.15),              # → indoor_commercial
            ("chemistry_lab", 0.05),         # → workspace_industrial
            ("biology_laboratory", 0.04),    # → workspace_industrial
        ]
        fn = _make_scene_fn_from_top1("pharmacy", 0.25, top5)
        out = fn(Image.new("RGB", (10, 10)))
        # 3/5 → indoor_commercial wins.
        assert out["scene_group"] == "indoor_commercial"

    def test_threshold_boundary_at_0_3(self):
        """Confidence exactly at 0.3 is NOT uncertain (>= threshold)."""
        top5 = [("kitchen", 0.30), ("dining_room", 0.20)]
        fn = _make_scene_fn_from_top1("kitchen", 0.30, top5)
        out = fn(Image.new("RGB", (10, 10)))
        assert out["category"] == "kitchen"

    def test_just_below_threshold_marks_uncertain(self):
        top5 = [("kitchen", 0.299), ("dining_room", 0.20)]
        fn = _make_scene_fn_from_top1("kitchen", 0.299, top5)
        out = fn(Image.new("RGB", (10, 10)))
        assert out["category"] == "uncertain"
