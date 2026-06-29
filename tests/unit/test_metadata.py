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
    apply_caption_merging,
    apply_ocr_merging,
    load_ocr_lookup,
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


# ── Task 3.1: OCR integration ──────────────────────────────────────────


def _write_ocr_csv(path: Path, rows: list[dict]) -> None:
    """Write a minimal ocr_results.csv with the pipeline's schema."""
    import csv as _csv

    fieldnames = [
        "video", "file", "path", "text_full", "num_boxes",
        "conf_mean", "boxes", "texts", "confs",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _boxes_str(polygons: list[list[list[float]]]) -> str:
    """Render a JSON list of 4-corner polygons (the pipeline's `boxes` format)."""
    import json as _json
    return _json.dumps(polygons, ensure_ascii=False)


def _texts_str(texts: list[str]) -> str:
    import json as _json
    return _json.dumps(texts, ensure_ascii=False)


def _confs_str(confs: list[float]) -> str:
    import json as _json
    return _json.dumps(confs, ensure_ascii=False)


class TestPolygonToXyxy:
    """Cover the polygon → xyxy conversion used by load_ocr_lookup."""

    def test_top_left_origin(self):
        from aic26.pipeline.metadata import _polygon_to_xyxy

        poly = [[10.0, 20.0], [110.0, 20.0], [110.0, 120.0], [10.0, 120.0]]
        assert _polygon_to_xyxy(poly) == [10.0, 20.0, 110.0, 120.0]

    def test_polygon_in_reading_order_still_yields_bounding_box(self):
        """The polygon is axis-aligned here; bbox is the polygon itself."""
        from aic26.pipeline.metadata import _polygon_to_xyxy

        poly = [[100, 100], [200, 100], [200, 200], [100, 200]]
        assert _polygon_to_xyxy(poly) == [100.0, 100.0, 200.0, 200.0]

    def test_empty_polygon_returns_zeros(self):
        from aic26.pipeline.metadata import _polygon_to_xyxy

        assert _polygon_to_xyxy([]) == [0.0, 0.0, 0.0, 0.0]

    def test_non_polygon_input_returns_zeros(self):
        """Non-list-of-points (string, None) falls back to zeros."""
        from aic26.pipeline.metadata import _polygon_to_xyxy

        assert _polygon_to_xyxy("not a polygon") == [0.0, 0.0, 0.0, 0.0]  # type: ignore[arg-type]
        assert _polygon_to_xyxy(None) == [0.0, 0.0, 0.0, 0.0]  # type: ignore[arg-type]

    def test_single_point_yields_zero_area_box(self):
        """A 1-point polygon is degenerate but still works (min == max)."""
        from aic26.pipeline.metadata import _polygon_to_xyxy

        assert _polygon_to_xyxy([[1, 2]]) == [1.0, 2.0, 1.0, 2.0]


class TestLoadOcrLookup:
    """`load_ocr_lookup` parses data/ocr_results.csv correctly."""

    def test_parses_csv_with_polygons(self, tmp_path: Path):
        csv = tmp_path / "ocr.csv"
        _write_ocr_csv(csv, [
            {
                "video": "K05_V004", "file": "001.jpg",
                "path": "data/keyframes/K05_V004/001.jpg",
                "text_full": "06:30", "num_boxes": 1,
                "conf_mean": 0.95,
                "boxes": _boxes_str([[[1580, 138], [1708, 138], [1708, 164], [1580, 164]]]),
                "texts": _texts_str(["06:30"]),
                "confs": _confs_str([0.95]),
            },
            {
                "video": "K05_V004", "file": "002.jpg",
                "path": "data/keyframes/K05_V004/002.jpg",
                "text_full": "EXIT", "num_boxes": 1,
                "conf_mean": 0.88,
                "boxes": _boxes_str([[[100, 200], [180, 200], [180, 240], [100, 240]]]),
                "texts": _texts_str(["EXIT"]),
                "confs": _confs_str([0.88]),
            },
        ])
        lookup = load_ocr_lookup(csv)
        assert ("K05_V004", "001") in lookup
        assert ("K05_V004", "002") in lookup
        # Polygon → xyxy
        rec_001 = lookup[("K05_V004", "001")][0]
        assert rec_001["text"] == "06:30"
        assert rec_001["bbox"] == [1580.0, 138.0, 1708.0, 164.0]
        assert rec_001["confidence"] == 0.95
        rec_002 = lookup[("K05_V004", "002")][0]
        assert rec_002["text"] == "EXIT"
        assert rec_002["bbox"] == [100.0, 200.0, 180.0, 240.0]

    def test_skips_empty_rows(self, tmp_path: Path):
        csv = tmp_path / "ocr.csv"
        _write_ocr_csv(csv, [
            {
                "video": "K05_V004", "file": "001.jpg", "path": "x",
                "text_full": "", "num_boxes": 0,
                "conf_mean": 0.0,
                "boxes": "[]", "texts": "[]", "confs": "[]",
            },
            {
                "video": "K05_V004", "file": "002.jpg", "path": "x",
                "text_full": "EXIT", "num_boxes": 1,
                "conf_mean": 0.9,
                "boxes": _boxes_str([[[0, 0], [50, 0], [50, 20], [0, 20]]]),
                "texts": _texts_str(["EXIT"]),
                "confs": _confs_str([0.9]),
            },
        ])
        lookup = load_ocr_lookup(csv)
        assert ("K05_V004", "001") not in lookup
        assert ("K05_V004", "002") in lookup

    def test_missing_csv_returns_empty_lookup(self, tmp_path: Path):
        lookup = load_ocr_lookup(tmp_path / "nope.csv")
        assert lookup == {}

    def test_handles_multiple_boxes_per_frame(self, tmp_path: Path):
        csv = tmp_path / "ocr.csv"
        _write_ocr_csv(csv, [
            {
                "video": "V", "file": "007.jpg", "path": "x",
                "text_full": "A B", "num_boxes": 2,
                "conf_mean": 0.8,
                "boxes": _boxes_str([
                    [[10, 10], [50, 10], [50, 30], [10, 30]],
                    [[60, 10], [100, 10], [100, 30], [60, 30]],
                ]),
                "texts": _texts_str(["A", "B"]),
                "confs": _confs_str([0.7, 0.9]),
            },
        ])
        lookup = load_ocr_lookup(csv)
        records = lookup[("V", "007")]
        assert len(records) == 2
        assert records[0]["text"] == "A"
        assert records[1]["text"] == "B"
        assert records[0]["bbox"] == [10.0, 10.0, 50.0, 30.0]
        assert records[1]["bbox"] == [60.0, 10.0, 100.0, 30.0]


class TestKeyframeMetadataOcrField:
    """`ocr_text` field round-trips and is omitted from to_dict when None."""

    def test_round_trip_preserves_ocr_text(self):
        m = KeyframeMetadata(
            video="V", frame_idx=42,
            detections=[],
            scene=None,
            scene_graph=None,
            ocr_text=[{"text": "EXIT", "bbox": [0, 0, 50, 20], "confidence": 0.95}],
        )
        s = m.to_json()
        loaded = KeyframeMetadata.from_json(s)
        assert loaded.ocr_text == [
            {"text": "EXIT", "bbox": [0, 0, 50, 20], "confidence": 0.95}
        ]

    def test_omitted_when_none(self):
        m = KeyframeMetadata(
            video="V", frame_idx=42,
            detections=[],
            scene=None,
            scene_graph=None,
        )
        d = m.to_dict()
        assert "ocr_text" not in d  # backward-compatible: no key when None

    def test_emitted_as_empty_list_when_explicitly_empty(self):
        """An empty list means 'OCR ran, no text'. A None means 'not applied'."""
        m = KeyframeMetadata(
            video="V", frame_idx=42,
            detections=[],
            scene=None,
            scene_graph=None,
            ocr_text=[],
        )
        d = m.to_dict()
        assert d["ocr_text"] == []

    def test_loaded_from_dict_with_no_ocr_key(self):
        """Old sidecars without ocr_text load cleanly with ocr_text=None."""
        m = KeyframeMetadata.from_dict({
            "video": "V", "frame_idx": 42,
            "detections": [], "scene": None, "scene_graph": None,
        })
        assert m.ocr_text is None


class TestApplyOcrMerging:
    """`apply_ocr_merging` writes ocr_text to sidecars on disk."""

    def test_writes_ocr_text_to_matching_sidecar(self, tmp_path: Path):
        # Set up a video dir with 2 sidecars.
        vdir = tmp_path / "V"
        vdir.mkdir()
        for stem in ("001", "002"):
            m = KeyframeMetadata(
                video="V", frame_idx=int(stem),
                detections=[], scene=None, scene_graph=None,
            )
            m.save(vdir / f"{stem}.json")

        # Build OCR lookup.
        ocr = {("V", "001"): [{"text": "EXIT", "bbox": [0, 0, 50, 20], "confidence": 0.9}]}
        stats = apply_ocr_merging(
            metadata_dir=tmp_path,
            ocr_lookup=ocr,
            video_name="V",
            overwrite_files=True,
        )
        assert stats["frames_total"] == 2
        assert stats["frames_with_text"] == 1
        assert stats["frames_empty"] == 1

        # Verify 001.json has ocr_text, 002.json has empty list.
        m1 = KeyframeMetadata.load(vdir / "001.json")
        m2 = KeyframeMetadata.load(vdir / "002.json")
        assert m1.ocr_text == [
            {"text": "EXIT", "bbox": [0, 0, 50, 20], "confidence": 0.9}
        ]
        assert m2.ocr_text == []  # ran, no text

    def test_dry_run_does_not_write(self, tmp_path: Path):
        vdir = tmp_path / "V"
        vdir.mkdir()
        m = KeyframeMetadata(
            video="V", frame_idx=1, detections=[], scene=None, scene_graph=None,
        )
        m.save(vdir / "001.json")
        ocr = {("V", "001"): [{"text": "X", "bbox": [0, 0, 1, 1], "confidence": 0.5}]}

        # Capture pre-state mtime.
        pre_path = vdir / "001.json"
        pre = pre_path.read_text()
        apply_ocr_merging(
            metadata_dir=tmp_path,
            ocr_lookup=ocr,
            video_name="V",
            overwrite_files=False,  # simulate "dry run" by not writing
        )
        # File untouched (still no ocr_text key).
        assert pre_path.read_text() == pre
        loaded = KeyframeMetadata.load(pre_path)
        assert loaded.ocr_text is None

    def test_aggregate_stats_counts_ocr(self, tmp_path: Path):
        vdir = tmp_path / "V"
        vdir.mkdir()
        for stem, ocr in [
            ("001", [{"text": "A", "bbox": [0, 0, 1, 1], "confidence": 0.5}]),
            ("002", []),  # ran, no text
        ]:
            m = KeyframeMetadata(
                video="V", frame_idx=int(stem),
                detections=[], scene=None, scene_graph=None,
                ocr_text=ocr,
            )
            m.save(vdir / f"{stem}.json")
        # Add one without ocr_text at all.
        m = KeyframeMetadata(
            video="V", frame_idx=3, detections=[], scene=None, scene_graph=None,
        )
        m.save(vdir / "003.json")

        stats = aggregate_metadata_stats(tmp_path)
        assert stats.sidecars_with_ocr == 2
        assert stats.ocr_total_text_regions == 1  # 1 record on frame 001 only


# ── Task 3.2: Caption field on KeyframeMetadata ────────────────────────


class TestKeyframeMetadataCaptionField:
    """`caption` field round-trips and is omitted from to_dict when None."""

    def test_round_trip_preserves_caption(self):
        m = KeyframeMetadata(
            video="V", frame_idx=42,
            detections=[], scene=None, scene_graph=None,
            caption="a black cat on a mat",
        )
        loaded = KeyframeMetadata.from_json(m.to_json())
        assert loaded.caption == "a black cat on a mat"

    def test_omitted_when_none(self):
        m = KeyframeMetadata(
            video="V", frame_idx=42,
            detections=[], scene=None, scene_graph=None,
        )
        d = m.to_dict()
        assert "caption" not in d  # backward-compatible: no key when None

    def test_emitted_as_empty_string_when_explicitly_empty(self):
        """Empty string means 'captioning ran, no output' — distinguishable
        from 'not yet applied' (None)."""
        m = KeyframeMetadata(
            video="V", frame_idx=42,
            detections=[], scene=None, scene_graph=None,
            caption="",
        )
        d = m.to_dict()
        assert d["caption"] == ""

    def test_loaded_from_dict_with_no_caption_key(self):
        """Old sidecars without `caption` load cleanly with caption=None."""
        m = KeyframeMetadata.from_dict({
            "video": "V", "frame_idx": 42,
            "detections": [], "scene": None, "scene_graph": None,
        })
        assert m.caption is None


# ── Task 3.2: MetadataExtractor caption plumbing ───────────────────────


class TestProcessKeyframeCaption:
    """MetadataExtractor wires caption_fn through process_keyframe."""

    def test_runs_caption_fn_when_provided(self, sample_image: Path):
        caption_fn = lambda img: "a sunset over the hills"  # noqa: E731
        ext = MetadataExtractor(
            detector_fn=_stub_detector,
            scene_fn=_stub_scene,
            scene_graph_fn=_stub_scene_graph,
            caption_fn=caption_fn,
        )
        meta = ext.process_keyframe(sample_image, video="L01_V001", frame_idx=1)
        assert meta is not None
        assert meta.caption == "a sunset over the hills"

    def test_caption_none_when_caption_fn_none(self, sample_image: Path):
        ext = MetadataExtractor(
            detector_fn=_stub_detector,
            scene_fn=_stub_scene,
            scene_graph_fn=_stub_scene_graph,
            # caption_fn defaults to None
        )
        meta = ext.process_keyframe(sample_image, video="L01_V001", frame_idx=1)
        assert meta.caption is None

    def test_caption_empty_string_when_model_returns_empty(self, sample_image: Path):
        ext = MetadataExtractor(
            detector_fn=_stub_detector,
            scene_fn=_stub_scene,
            scene_graph_fn=_stub_scene_graph,
            caption_fn=lambda img: "",
        )
        meta = ext.process_keyframe(sample_image, video="L01_V001", frame_idx=1)
        # Empty string preserved (distinguishable from None).
        assert meta.caption == ""

    def test_skip_caption_flag(self, sample_image: Path):
        caption_fn = lambda img: "should not run"  # noqa: E731
        ext = MetadataExtractor(
            detector_fn=_stub_detector,
            scene_fn=_stub_scene,
            scene_graph_fn=_stub_scene_graph,
            caption_fn=caption_fn,
        )
        meta = ext.process_keyframe(
            sample_image, video="L01_V001", frame_idx=1, skip_caption=True,
        )
        assert meta.caption is None  # skip_caption short-circuits

    def test_caption_fn_exception_yields_none(self, sample_image: Path):
        def bad_fn(img):
            raise RuntimeError("CUDA OOM")

        ext = MetadataExtractor(
            detector_fn=_stub_detector,
            scene_fn=_stub_scene,
            scene_graph_fn=_stub_scene_graph,
            caption_fn=bad_fn,
        )
        meta = ext.process_keyframe(sample_image, video="L01_V001", frame_idx=1)
        # Graceful degradation: None, not exception.
        assert meta.caption is None


# ── Task 3.2: apply_caption_merging post-pass ──────────────────────────


class TestApplyCaptionMerging:
    """apply_caption_merging() writes caption field to sidecars in-place."""

    def _make_keyframe_jpg(self, parent: Path, stem: str) -> Path:
        """Create a tiny valid JPEG for the keyframe."""
        from PIL import Image as _PILImage

        parent.mkdir(parents=True, exist_ok=True)
        img = _PILImage.new("RGB", (8, 8), color=(128, 128, 128))
        path = parent / f"{stem}.jpg"
        img.save(path)
        return path

    def _write_sidecar(self, video_dir: Path, stem: str, *, caption: str | None = None):
        m = KeyframeMetadata(
            video=video_dir.name, frame_idx=int(stem),
            detections=[], scene=None, scene_graph=None,
            caption=caption,
        )
        m.save(video_dir / f"{stem}.json")

    def test_writes_caption_to_matching_sidecar(self, tmp_path: Path):
        vdir = tmp_path / "V"
        vdir.mkdir()
        # Create matching keyframes + sidecars.
        for stem in ("001", "002"):
            self._make_keyframe_jpg(tmp_path / "keyframes" / "V", stem)
            self._write_sidecar(vdir, stem)

        kf_dir = tmp_path / "keyframes"

        def stub_caption(img):
            return "a cat on a mat"

        stats = apply_caption_merging(
            metadata_dir=tmp_path,
            caption_fn=stub_caption,
            keyframes_dir=kf_dir,
            video_name="V",
            overwrite_files=True,
        )
        assert stats["frames_total"] == 2
        assert stats["frames_captioned"] == 2
        assert stats["frames_empty"] == 0

        m1 = KeyframeMetadata.load(vdir / "001.json")
        m2 = KeyframeMetadata.load(vdir / "002.json")
        assert m1.caption == "a cat on a mat"
        assert m2.caption == "a cat on a mat"

    def test_dry_run_does_not_write(self, tmp_path: Path):
        vdir = tmp_path / "V"
        vdir.mkdir()
        self._make_keyframe_jpg(tmp_path / "keyframes" / "V", "001")
        self._write_sidecar(vdir, "001")  # caption=None initially

        pre = (vdir / "001.json").read_text()

        apply_caption_merging(
            metadata_dir=tmp_path,
            caption_fn=lambda img: "a fresh caption",
            keyframes_dir=tmp_path / "keyframes",
            video_name="V",
            overwrite_files=False,  # simulate "dry run"
        )
        # Sidecar untouched.
        assert (vdir / "001.json").read_text() == pre
        loaded = KeyframeMetadata.load(vdir / "001.json")
        assert loaded.caption is None

    def test_skips_sidecars_with_existing_caption(self, tmp_path: Path):
        vdir = tmp_path / "V"
        vdir.mkdir()
        self._make_keyframe_jpg(tmp_path / "keyframes" / "V", "001")
        # Sidecar already has a caption.
        self._write_sidecar(vdir, "001", caption="old caption")

        stats = apply_caption_merging(
            metadata_dir=tmp_path,
            caption_fn=lambda img: "new caption",
            keyframes_dir=tmp_path / "keyframes",
            video_name="V",
            overwrite_files=False,
        )
        assert stats["frames_skipped"] == 1
        # Old caption preserved.
        loaded = KeyframeMetadata.load(vdir / "001.json")
        assert loaded.caption == "old caption"

    def test_handles_missing_keyframe(self, tmp_path: Path):
        vdir = tmp_path / "V"
        vdir.mkdir()
        # No keyframe image exists — only a sidecar.
        self._write_sidecar(vdir, "001")

        stats = apply_caption_merging(
            metadata_dir=tmp_path,
            caption_fn=lambda img: "should not run",
            keyframes_dir=tmp_path / "keyframes",  # doesn't exist
            video_name="V",
            overwrite_files=True,
        )
        assert stats["frames_failed"] == 1
        # Sidecar should not have been modified.
        loaded = KeyframeMetadata.load(vdir / "001.json")
        assert loaded.caption is None

    def test_aggregate_stats_counts_caption(self, tmp_path: Path):
        vdir = tmp_path / "V"
        vdir.mkdir()
        for stem, caption in [
            ("001", "a cat"),
            ("002", ""),  # ran, no output
        ]:
            self._write_sidecar(vdir, stem, caption=caption)
        # Sidecar without caption.
        self._write_sidecar(vdir, "003")

        stats = aggregate_metadata_stats(tmp_path)
        assert stats.sidecars_with_caption == 2  # 001 and 002 ("" counts)
        assert stats.caption_total_chars == len("a cat") + len("")  # 5 + 0
