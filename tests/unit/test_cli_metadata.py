"""
Unit tests for the metadata CLI subcommands under `aic26 pipeline`.

These exercise the Click command handlers, not the Click invocation
machinery. We test the underlying functions directly by extracting them
from the cli module. This avoids spawning a subprocess for each test
and keeps the suite fast.

What we cover:
1. `pipeline extract-metadata` invokes the orchestrator with the right
   args (skip flags, paths, overwrite).
2. `pipeline verify-metadata` reads one sidecar and prints it.
3. `pipeline metadata-stats` calls aggregate_metadata_stats and formats.
4. `run-all` has a `--extract-metadata` flag (AD-5 opt-in).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from PIL import Image

from aic26.cli.main import cli
from aic26.pipeline.metadata import KeyframeMetadata

# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def kf_root_with_video(tmp_path: Path) -> Path:
    kf = tmp_path / "keyframes" / "L01_V001"
    kf.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (32, 32), color=(i, i, i)).save(kf / f"frame_{i:04d}.jpg")
    return tmp_path / "keyframes"


@pytest.fixture
def existing_metadata_root(tmp_path: Path) -> Path:
    root = tmp_path / "metadata"
    vdir = root / "L01_V001"
    vdir.mkdir(parents=True)
    KeyframeMetadata(
        video="L01_V001",
        frame_idx=0,
        detections=[{"label": "Person", "score": 0.9, "bbox": [1, 2, 3, 4], "position": "center"}],
        scene={"category": "kitchen", "confidence": 0.85, "top5": [["kitchen", 0.85]]},
        scene_graph=[{"s": "person", "p": "hold", "o": "cup"}],
    ).save(vdir / "frame_0000.json")
    return root


# ── extract-metadata subcommand ───────────────────────────────────────


class TestExtractMetadataCmd:
    def test_subcommand_registered(self, runner: CliRunner):
        result = runner.invoke(cli, ["pipeline", "--help"])
        assert result.exit_code == 0
        assert "extract-metadata" in result.output

    def test_invokes_orchestrator(
        self,
        runner: CliRunner,
        kf_root_with_video: Path,
        tmp_path: Path,
    ):
        out = tmp_path / "metadata"
        with patch("aic26.pipeline.metadata.MetadataExtractor") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.process_all.return_value = {
                "videos_total": 1,
                "videos_processed": 1,
                "frames_total": 3,
                "frames_processed": 3,
                "frames_skipped": 0,
                "frames_failed": 0,
            }
            mock_cls.return_value = mock_instance
            result = runner.invoke(
                cli,
                [
                    "pipeline",
                    "extract-metadata",
                    "--input",
                    str(kf_root_with_video),
                    "--output",
                    str(out),
                ],
            )
            assert result.exit_code == 0, result.output
            assert mock_cls.called
            assert mock_instance.process_all.called

    def test_skip_flags_passed_through(
        self,
        runner: CliRunner,
        kf_root_with_video: Path,
        tmp_path: Path,
    ):
        out = tmp_path / "metadata"
        with patch("aic26.pipeline.metadata.MetadataExtractor") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.process_all.return_value = {
                "videos_total": 1,
                "videos_processed": 1,
                "frames_total": 3,
                "frames_processed": 3,
                "frames_skipped": 0,
                "frames_failed": 0,
            }
            mock_cls.return_value = mock_instance
            result = runner.invoke(
                cli,
                [
                    "pipeline",
                    "extract-metadata",
                    "--input",
                    str(kf_root_with_video),
                    "--output",
                    str(out),
                    "--skip-scene",
                    "--skip-scene-graph",
                ],
            )
            assert result.exit_code == 0, result.output
            call_kwargs = mock_instance.process_all.call_args.kwargs
            assert call_kwargs.get("skip_scene") is True
            assert call_kwargs.get("skip_scene_graph") is True

    def test_skip_detections_flag(self, runner: CliRunner, kf_root_with_video, tmp_path):
        out = tmp_path / "metadata"
        with patch("aic26.pipeline.metadata.MetadataExtractor") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.process_all.return_value = {
                "videos_total": 0,
                "videos_processed": 0,
                "frames_total": 0,
                "frames_processed": 0,
                "frames_skipped": 0,
                "frames_failed": 0,
            }
            mock_cls.return_value = mock_instance
            result = runner.invoke(
                cli,
                [
                    "pipeline",
                    "extract-metadata",
                    "--input",
                    str(kf_root_with_video),
                    "--output",
                    str(out),
                    "--skip-detections",
                ],
            )
            assert result.exit_code == 0, result.output
            call_kwargs = mock_instance.process_all.call_args.kwargs
            assert call_kwargs.get("skip_detections") is True

    def test_overwrite_flag(self, runner: CliRunner, kf_root_with_video, tmp_path):
        out = tmp_path / "metadata"
        with patch("aic26.pipeline.metadata.MetadataExtractor") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.process_all.return_value = {
                "videos_total": 1,
                "videos_processed": 1,
                "frames_total": 3,
                "frames_processed": 3,
                "frames_skipped": 0,
                "frames_failed": 0,
            }
            mock_cls.return_value = mock_instance
            result = runner.invoke(
                cli,
                [
                    "pipeline",
                    "extract-metadata",
                    "--input",
                    str(kf_root_with_video),
                    "--output",
                    str(out),
                    "--overwrite",
                ],
            )
            assert result.exit_code == 0, result.output
            call_kwargs = mock_instance.process_all.call_args.kwargs
            assert call_kwargs.get("overwrite") is True

    def test_missing_input_fails(self, runner: CliRunner, tmp_path: Path):
        result = runner.invoke(
            cli,
            [
                "pipeline",
                "extract-metadata",
                "--input",
                str(tmp_path / "does-not-exist"),
                "--output",
                str(tmp_path / "metadata"),
            ],
        )
        assert result.exit_code != 0


# ── verify-metadata subcommand ────────────────────────────────────────


class TestVerifyMetadataCmd:
    def test_subcommand_registered(self, runner: CliRunner):
        result = runner.invoke(cli, ["pipeline", "--help"])
        assert "verify-metadata" in result.output

    def test_loads_and_prints_sidecar(
        self,
        runner: CliRunner,
        existing_metadata_root: Path,
    ):
        result = runner.invoke(
            cli,
            [
                "pipeline",
                "verify-metadata",
                "--input",
                str(existing_metadata_root),
                "--video",
                "L01_V001",
                "--frame-idx",
                "0",
            ],
        )
        assert result.exit_code == 0, result.output
        # Sidecar content includes category and triple
        assert "kitchen" in result.output
        assert "person" in result.output

    def test_missing_video_exits_with_error(
        self,
        runner: CliRunner,
        existing_metadata_root: Path,
    ):
        result = runner.invoke(
            cli,
            [
                "pipeline",
                "verify-metadata",
                "--input",
                str(existing_metadata_root),
                "--video",
                "MISSING",
                "--frame-idx",
                "0",
            ],
        )
        assert result.exit_code != 0

    def test_missing_frame_exits_with_error(
        self,
        runner: CliRunner,
        existing_metadata_root: Path,
    ):
        result = runner.invoke(
            cli,
            [
                "pipeline",
                "verify-metadata",
                "--input",
                str(existing_metadata_root),
                "--video",
                "L01_V001",
                "--frame-idx",
                "999",
            ],
        )
        assert result.exit_code != 0


# ── metadata-stats subcommand ──────────────────────────────────────────


class TestMetadataStatsCmd:
    def test_subcommand_registered(self, runner: CliRunner):
        result = runner.invoke(cli, ["pipeline", "--help"])
        assert "metadata-stats" in result.output

    def test_prints_stats(
        self,
        runner: CliRunner,
        existing_metadata_root: Path,
    ):
        result = runner.invoke(
            cli,
            ["pipeline", "metadata-stats", "--input", str(existing_metadata_root)],
        )
        assert result.exit_code == 0, result.output
        assert "sidecar" in result.output.lower() or "1" in result.output
        assert "kitchen" in result.output.lower()

    def test_missing_input_fails(self, runner: CliRunner, tmp_path: Path):
        result = runner.invoke(
            cli,
            ["pipeline", "metadata-stats", "--input", str(tmp_path / "does-not-exist")],
        )
        assert result.exit_code != 0


# ── AD-5: run-all --extract-metadata flag ─────────────────────────────


class TestRunAllExtractMetadataFlag:
    def test_run_all_runs_extract_metadata_when_flag_set(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        # Touch a dummy file so video-dir passes type check.
        (video_dir / "fake.mp4").write_bytes(b"\x00")

        # We patch the orchestrator so the test stays at the CLI level only.
        with patch("aic26.pipeline.orchestrator.PipelineOrchestrator") as mock_orch:
            orch = MagicMock()
            orch.run.return_value = {
                "_pipeline_stats": {
                    "steps_completed": ["keyframe_extraction", "metadata_extraction"],
                    "steps_failed": [],
                    "steps_skipped": [],
                    "total_duration_seconds": 1.0,
                },
            }
            mock_orch.return_value = orch

            result = runner.invoke(
                cli,
                ["pipeline", "run-all", "--video-dir", str(video_dir), "--extract-metadata"],
            )
            assert result.exit_code == 0, result.output
            kwargs = orch.run.call_args.kwargs
            assert kwargs.get("extract_metadata") is True

    def test_run_all_does_not_run_metadata_by_default(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "fake.mp4").write_bytes(b"\x00")

        with patch("aic26.pipeline.orchestrator.PipelineOrchestrator") as mock_orch:
            orch = MagicMock()
            orch.run.return_value = {
                "_pipeline_stats": {
                    "steps_completed": ["keyframe_extraction"],
                    "steps_failed": [],
                    "steps_skipped": [],
                    "total_duration_seconds": 1.0,
                },
            }
            mock_orch.return_value = orch

            result = runner.invoke(
                cli,
                ["pipeline", "run-all", "--video-dir", str(video_dir)],
            )
            assert result.exit_code == 0, result.output
            kwargs = orch.run.call_args.kwargs
            # Default behavior: opt-in flag absent → flag is False
            assert kwargs.get("extract_metadata") is False


# ── Task 3.2: caption flag + merge-captions subcommand ──────────────────


class TestExtractMetadataCmdCaption:
    """Tests for the --skip-caption flag and caption_fn plumbing."""

    def test_skip_caption_flag(
        self, runner: CliRunner, kf_root_with_video: Path, tmp_path: Path
    ):
        out = tmp_path / "metadata"
        with patch("aic26.pipeline.metadata.MetadataExtractor") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.process_all.return_value = {
                "videos_total": 0, "videos_processed": 0,
                "frames_total": 0, "frames_processed": 0,
                "frames_skipped": 0, "frames_failed": 0,
            }
            mock_cls.return_value = mock_instance
            result = runner.invoke(
                cli,
                [
                    "pipeline", "extract-metadata",
                    "--input", str(kf_root_with_video),
                    "--output", str(out),
                    "--skip-caption",
                ],
            )
            assert result.exit_code == 0, result.output
            call_kwargs = mock_instance.process_all.call_args.kwargs
            assert call_kwargs.get("skip_caption") is True

    def test_passes_caption_fn_to_extractor(
        self, runner: CliRunner, kf_root_with_video: Path, tmp_path: Path
    ):
        """When --skip-caption is NOT set, build_metadata_callables should
        return a 4-tuple with a non-None caption_fn, and MetadataExtractor
        should receive it."""
        out = tmp_path / "metadata"
        # build_metadata_callables may attempt to load the model. We
        # patch the import to return a fake 4-tuple.
        fake_caption_fn = MagicMock(return_value="a cat")
        with patch(
            "aic26.pipeline.metadata.build_metadata_callables",
            return_value=(MagicMock(), MagicMock(), MagicMock(), fake_caption_fn),
        ), patch("aic26.pipeline.metadata.MetadataExtractor") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.process_all.return_value = {
                "videos_total": 0, "videos_processed": 0,
                "frames_total": 0, "frames_processed": 0,
                "frames_skipped": 0, "frames_failed": 0,
            }
            mock_cls.return_value = mock_instance
            result = runner.invoke(
                cli,
                [
                    "pipeline", "extract-metadata",
                    "--input", str(kf_root_with_video),
                    "--output", str(out),
                ],
            )
            assert result.exit_code == 0, result.output
            # The fake caption_fn was passed to the constructor.
            init_kwargs = mock_cls.call_args.kwargs
            assert init_kwargs.get("caption_fn") is fake_caption_fn


class TestMergeCaptionsCmd:
    """Tests for the new `pipeline merge-captions` subcommand."""

    def test_subcommand_registered(self, runner: CliRunner):
        result = runner.invoke(cli, ["pipeline", "--help"])
        assert result.exit_code == 0
        assert "merge-captions" in result.output

    def test_help_lists_required_options(self, runner: CliRunner):
        result = runner.invoke(cli, ["pipeline", "merge-captions", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--keyframes-dir" in result.output

    def test_invokes_apply_caption_merging(
        self, runner: CliRunner, tmp_path: Path
    ):
        # Create metadata dir with a fake sidecar and matching keyframe.
        vdir = tmp_path / "metadata_root" / "V"
        vdir.mkdir(parents=True)
        m = KeyframeMetadata(
            video="V", frame_idx=1,
            detections=[], scene=None, scene_graph=None,
        )
        m.save(vdir / "001.json")
        # Matching keyframe.
        from PIL import Image as _PILImage
        kf_dir = tmp_path / "kf" / "V"
        kf_dir.mkdir(parents=True)
        _PILImage.new("RGB", (8, 8), color=(0, 0, 0)).save(kf_dir / "001.jpg")

        with patch(
            "aic26.pipeline.metadata.apply_caption_merging",
            return_value={
                "frames_total": 1, "frames_captioned": 1, "frames_empty": 0,
                "frames_skipped": 0, "frames_failed": 0,
            },
        ) as mock_apply, patch(
            "aic26.pipeline.captioning.build_caption_fn",
            return_value=MagicMock(return_value="a cat"),
        ):
            result = runner.invoke(
                cli,
                [
                    "pipeline", "merge-captions",
                    "--input", str(tmp_path / "metadata_root"),
                    "--keyframes-dir", str(tmp_path / "kf"),
                    "--video", "V",
                ],
            )
            assert result.exit_code == 0, result.output
            assert mock_apply.called
            kwargs = mock_apply.call_args.kwargs
            assert kwargs.get("video_name") == "V"
            assert kwargs.get("overwrite_files") is True

    def test_dry_run_does_not_write(
        self, runner: CliRunner, tmp_path: Path
    ):
        vdir = tmp_path / "metadata_root" / "V"
        vdir.mkdir(parents=True)
        m = KeyframeMetadata(
            video="V", frame_idx=1,
            detections=[], scene=None, scene_graph=None,
        )
        m.save(vdir / "001.json")
        # Matching keyframe.
        from PIL import Image as _PILImage
        kf_dir = tmp_path / "kf" / "V"
        kf_dir.mkdir(parents=True)
        _PILImage.new("RGB", (8, 8), color=(0, 0, 0)).save(kf_dir / "001.jpg")

        with patch(
            "aic26.pipeline.metadata.apply_caption_merging",
            return_value={
                "frames_total": 1, "frames_captioned": 0, "frames_empty": 1,
                "frames_skipped": 0, "frames_failed": 0,
            },
        ) as mock_apply, patch(
            "aic26.pipeline.captioning.build_caption_fn",
            return_value=MagicMock(return_value=""),
        ):
            result = runner.invoke(
                cli,
                [
                    "pipeline", "merge-captions",
                    "--input", str(tmp_path / "metadata_root"),
                    "--keyframes-dir", str(tmp_path / "kf"),
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0, result.output
            # overwrite_files=False under dry-run.
            kwargs = mock_apply.call_args.kwargs
            assert kwargs.get("overwrite_files") is False
            # Sidecar still has no caption.
            loaded = KeyframeMetadata.load(vdir / "001.json")
            assert loaded.caption is None
