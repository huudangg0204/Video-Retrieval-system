"""
Smoke tests for scripts/smoke_test_metadata.py.

These tests verify the script's argument parsing and stub-mode execution
without loading any real ML model. The script is exercised by importing
its main() function directly and calling it with argv-like lists.

What we cover:
1. CLI arg parsing — required vs optional flags.
2. --video-dir missing → exit code 1 with stderr.
3. --video-dir pointing to an empty dir → exit code 1.
4. Stub-mode end-to-end run on a temp keyframe folder — exit code 0,
   writes a sidecar per keyframe, idempotent re-run skips.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

# Path setup so `scripts/smoke_test_metadata.py` can be imported as a module.
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import smoke_test_metadata as smoke  # type: ignore[import-not-found]  # noqa: E402


@pytest.fixture
def video_with_keyframes(tmp_path: Path) -> Path:
    vdir = tmp_path / "L01_V001"
    vdir.mkdir()
    for i in range(5):
        Image.new("RGB", (64, 36), color=(i * 30, 50, 100)).save(
            vdir / f"frame_{i:04d}.jpg",
        )
    return vdir


# ── Argument parsing ──────────────────────────────────────────────────


class TestParseArgs:
    def test_required_video_dir(self):
        # No args → SystemExit (argparse requirement)
        with pytest.raises(SystemExit):
            smoke.parse_args([])

    def test_defaults(self, video_with_keyframes: Path):
        args = smoke.parse_args(["--video-dir", str(video_with_keyframes)])
        assert args.video_dir == str(video_with_keyframes)
        assert args.output_dir == "data/metadata"
        assert args.limit == 50
        assert args.overwrite is False
        assert args.real_models is False
        assert args.skip_detections is False
        assert args.skip_scene is False
        assert args.skip_scene_graph is False

    def test_overwrite_and_skip_flags(self, video_with_keyframes: Path):
        args = smoke.parse_args(
            [
                "--video-dir",
                str(video_with_keyframes),
                "--overwrite",
                "--skip-detections",
                "--skip-scene-graph",
                "--limit",
                "10",
            ]
        )
        assert args.overwrite is True
        assert args.skip_detections is True
        assert args.skip_scene_graph is True
        assert args.skip_scene is False
        assert args.limit == 10


# ── Main flow (stub mode) ─────────────────────────────────────────────


class TestMain:
    def test_missing_video_dir_exits_1(self, tmp_path: Path):
        rc = smoke.main(["--video-dir", str(tmp_path / "does-not-exist")])
        assert rc == 1

    def test_empty_video_dir_exits_1(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        rc = smoke.main(["--video-dir", str(empty)])
        assert rc == 1

    def test_stub_mode_writes_sidecars(
        self,
        video_with_keyframes: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        out = tmp_path / "metadata"
        rc = smoke.main(
            [
                "--video-dir",
                str(video_with_keyframes),
                "--output-dir",
                str(out),
                "--limit",
                "3",
            ]
        )
        assert rc == 0
        # All 3 processed
        out_vdir = out / video_with_keyframes.name
        assert out_vdir.exists()
        sidecars = sorted(out_vdir.glob("*.json"))
        assert len(sidecars) == 3

        captured = capsys.readouterr()
        # Output mentions all keyframe files
        for i in range(3):
            assert f"frame_{i:04d}" in captured.out

    def test_stub_mode_idempotent(
        self,
        video_with_keyframes: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        out = tmp_path / "metadata"
        # First run: processes all 5
        smoke.main(
            [
                "--video-dir",
                str(video_with_keyframes),
                "--output-dir",
                str(out),
                "--limit",
                "5",
            ]
        )
        # Second run: skips all 5
        rc = smoke.main(
            [
                "--video-dir",
                str(video_with_keyframes),
                "--output-dir",
                str(out),
                "--limit",
                "5",
            ]
        )
        captured = capsys.readouterr()
        assert rc == 0
        assert "5" in captured.out  # frames_total
        assert "skipped" in captured.out.lower()

    def test_skip_detections_keeps_scene(
        self,
        video_with_keyframes: Path,
        tmp_path: Path,
    ):
        out = tmp_path / "metadata"
        smoke.main(
            [
                "--video-dir",
                str(video_with_keyframes),
                "--output-dir",
                str(out),
                "--limit",
                "1",
                "--skip-detections",
            ]
        )
        import json as json_mod

        sidecar = list((out / video_with_keyframes.name).glob("*.json"))[0]
        data = json_mod.loads(sidecar.read_text())
        assert data["detections"] == []
        assert data["scene"] is not None  # still ran
        assert data["scene"]["category"] == "kitchen"

    def test_skip_scene_graph_keeps_scene(
        self,
        video_with_keyframes: Path,
        tmp_path: Path,
    ):
        out = tmp_path / "metadata"
        smoke.main(
            [
                "--video-dir",
                str(video_with_keyframes),
                "--output-dir",
                str(out),
                "--limit",
                "1",
                "--skip-scene-graph",
            ]
        )
        import json as json_mod

        sidecar = list((out / video_with_keyframes.name).glob("*.json"))[0]
        data = json_mod.loads(sidecar.read_text())
        assert data["scene_graph"] is None
        assert data["scene"] is not None


# ── Stub callable shape ───────────────────────────────────────────────


class TestStubs:
    def test_stub_detector_returns_yolo_shape(self, tmp_path: Path):
        img_path = tmp_path / "frame_0001.jpg"
        Image.new("RGB", (32, 32)).save(img_path)
        result = smoke._stub_detector(img_path)
        assert result is not None
        assert "detection_scores" in result
        assert "detection_boxes" in result
        assert "detection_positions" in result

    def test_stub_scene_returns_places_shape(self):
        result = smoke._stub_scene(None)
        assert result["category"] == "kitchen"
        assert isinstance(result["top5"], list)
        assert len(result["top5"]) == 5

    def test_stub_scene_graph_returns_triple_dicts(self):
        result = smoke._stub_scene_graph(None)
        assert isinstance(result, list)
        assert result == [{"s": "person", "p": "hold", "o": "cup"}]
