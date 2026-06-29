"""
Tests for `Places365Classifier.from_weights()` and the model loader.

These tests exercise the wiring that turns a downloaded
`resnet50_places365.pth.tar` into a working classifier. They avoid
loading 365 real labels by stubbing `load_places365_labels`.

Covers:
1. from_weights raises FileNotFoundError on missing weights.
2. from_weights rejects non-dict checkpoints.
3. from_weights rejects unsupported arch fields.
4. _strip_module_prefix removes the DataParallel prefix on every key.
5. _build_resnet50_places365 returns a ResNet50 with fc out_features=365.
6. _build_model_from_state_dict wires the FC head even when arch differs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SCRIPTS_DIR))

import aic26.pipeline.places365 as p365  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────


class _TinyLabels:
    """Stubbed labels loader — returns a 365-entry list of 'cat_N' strings."""

    @staticmethod
    def fake(labels_path):  # noqa: ARG001
        return [f"cat_{i}" for i in range(365)]


@pytest.fixture
def fake_resnet50_pth(tmp_path: Path) -> Path:
    """Build a tiny ResNet50 state dict and write a .pth.tar matching CSAIL format."""
    from torchvision.models import resnet50

    real = resnet50(weights=None, num_classes=365)
    sd = real.state_dict()
    # Mimic DataParallel: prefix every key with "module."
    prefixed = {f"module.{k}": v for k, v in sd.items()}
    p = tmp_path / "tiny_resnet50.pth.tar"
    torch.save(
        {
            "state_dict": prefixed,
            "epoch": 0,
            "arch": "resnet50",
            "best_prec1": 0.0,
        },
        p,
    )
    return p


# ── load_places365_labels format parsing ──────────────────────────────


class TestLoadLabels:
    def test_csail_format_with_trailing_id(self, tmp_path: Path):
        f = tmp_path / "categories_places365.txt"
        f.write_text(
            "/a/airfield 0\n"
            "/a/airplane_cabin 1\n"
            "/a/airport_terminal 2\n"
            "/y/yard 362\n"
            "/z/zen_garden 364\n"
        )
        labels = p365.load_places365_labels(f)
        assert labels == ["airfield", "airplane_cabin", "airport_terminal", "yard", "zen_garden"]

    def test_legacy_format_without_id(self, tmp_path: Path):
        f = tmp_path / "categories_places365.txt"
        f.write_text("/a/kitchen/kitchen\n/a/yard/yard\n")
        labels = p365.load_places365_labels(f)
        assert labels == ["kitchen", "yard"]

    def test_lowercases(self, tmp_path: Path):
        f = tmp_path / "categories_places365.txt"
        f.write_text("/a/Beach_Beach 5\n")
        labels = p365.load_places365_labels(f)
        assert labels == ["beach_beach"]

    def test_skips_blank_lines(self, tmp_path: Path):
        f = tmp_path / "categories_places365.txt"
        f.write_text("\n\n/a/yard/yard\n\n\n")
        labels = p365.load_places365_labels(f)
        assert labels == ["yard"]

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            p365.load_places365_labels(tmp_path / "nope.txt")


# ── from_weights error paths ──────────────────────────────────────────


class TestFromWeightsErrors:
    def test_missing_weights_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            p365.Places365Classifier.from_weights(
                weights_path=tmp_path / "nope.pth.tar",
                labels_path=tmp_path / "labels.txt",
            )

    def test_non_dict_checkpoint_raises(self, tmp_path: Path):
        p = tmp_path / "weird.pth.tar"
        torch.save([1, 2, 3], p)  # list, not dict
        with pytest.raises(ValueError, match="Unexpected"):
            p365.Places365Classifier.from_weights(
                weights_path=p,
                labels_path=tmp_path / "labels.txt",
            )

    def test_unsupported_arch_raises(self, tmp_path: Path):
        p = tmp_path / "vgg.pth.tar"
        torch.save({"state_dict": {}, "arch": "vgg16"}, p)
        with pytest.raises(ValueError, match="Unsupported"):
            p365.Places365Classifier.from_weights(
                weights_path=p,
                labels_path=tmp_path / "labels.txt",
            )


# ── _strip_module_prefix ──────────────────────────────────────────────


class TestStripPrefix:
    def test_removes_module_prefix(self):
        sd = {
            "module.conv1.weight": torch.zeros(1),
            "module.bn1.weight": torch.zeros(1),
            "fc.weight": torch.zeros(1),
            "no_prefix_key": torch.zeros(1),
        }
        result = p365._strip_module_prefix(sd)
        assert set(result.keys()) == {"conv1.weight", "bn1.weight", "fc.weight", "no_prefix_key"}
        # "module." removed cleanly, no double-strip
        assert "module.conv1.weight" not in result

    def test_empty(self):
        assert p365._strip_module_prefix({}) == {}


# ── _build_resnet50_places365 ─────────────────────────────────────────


class TestBuildResNet50:
    def test_365_class_head(self):
        model = p365._build_resnet50_places365()
        assert isinstance(model.fc, nn.Linear)
        assert model.fc.out_features == 365
        assert model.fc.in_features == 2048


# ── _build_model_from_state_dict ──────────────────────────────────────


class TestBuildFromStateDict:
    def test_wires_resnet50(self, fake_resnet50_pth: Path):
        sd = torch.load(fake_resnet50_pth, map_location="cpu", weights_only=False)
        model = p365._build_model_from_state_dict(sd)
        assert isinstance(model, nn.Module)
        # FC head has the right output dim
        assert model.fc.out_features == 365

    def test_strict_false_tolerates_fc_mismatch(self):
        # SD with FC shape [100, 2048] — should still load, FC reinit.
        from torchvision.models import resnet50

        base = resnet50(weights=None, num_classes=100)
        sd = base.state_dict()
        prefixed = {f"module.{k}": v for k, v in sd.items()}
        wrapped = {"state_dict": prefixed, "arch": "resnet50"}
        # Should NOT raise; model.fc must end up with 365 outputs.
        model = p365._build_model_from_state_dict(wrapped)
        assert model.fc.out_features == 365


# ── from_weights happy path (stubbed labels loader) ──────────────────


class TestFromWeightsHappy:
    def test_constructs_classifier(self, tmp_path: Path, fake_resnet50_pth: Path):
        labels = tmp_path / "labels.txt"
        labels.write_text("/a/kitchen/kitchen\n")
        with patch.object(p365, "load_places365_labels", _TinyLabels.fake):
            clf = p365.Places365Classifier.from_weights(
                weights_path=fake_resnet50_pth,
                labels_path=labels,
                device="cpu",
            )
            assert clf._model is not None
            # Labels resolved via the patched loader
            assert len(clf.labels) == 365
            assert clf.device == "cpu"

    def test_classify_returns_topk(self, tmp_path: Path, fake_resnet50_pth: Path):
        from PIL import Image

        labels = tmp_path / "labels.txt"
        labels.write_text("/a/kitchen/kitchen\n")
        with patch.object(p365, "load_places365_labels", _TinyLabels.fake):
            clf = p365.Places365Classifier.from_weights(
                weights_path=fake_resnet50_pth,
                labels_path=labels,
                device="cpu",
                top_k=3,
            )
            # Random image — should still produce 3 top categories.
            img = Image.new("RGB", (256, 256), color=(127, 64, 200))
            result = clf.classify(img)
            assert result.category.startswith("cat_")
            assert 0.0 <= result.confidence <= 1.0
            assert len(result.top5) == 3
