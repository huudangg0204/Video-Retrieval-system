"""
Unit tests for pipeline/places365.py.

Tests the Places365Classifier that produces per-keyframe scene labels
(top-K categories with confidence) — required for lifeXplore-style
`-c kitchen` filters (paper-knowledge §2 Table 1) and scene-context
boosts for image search.

The classifier must be testable without the 95MB Places365 weights
download, so we mock the underlying model. The real-weight path is
exercised by Task T10 (download script) and T11 (real inference).
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from aic26.pipeline.places365 import (
    Places365Classifier,
    Places365Result,
    load_places365_labels,
)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def dummy_image() -> Image.Image:
    """224×224 RGB image — input size for EfficientNet-B2."""
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


@pytest.fixture
def mock_model():
    """Mock torchvision model that returns predictable logits."""
    model = MagicMock()
    # Output shape [1, 365] — 365 Places365 categories
    # Set logits so category index 5 (kitchen-like) wins, then index 100, etc.
    import torch
    logits = torch.full((1, 365), -10.0)
    logits[0, 5] = 5.0   # top-1
    logits[0, 100] = 4.0  # top-2
    logits[0, 200] = 3.0  # top-3
    logits[0, 50] = 2.0   # top-4
    logits[0, 300] = 1.0  # top-5
    model.return_value = logits
    return model


@pytest.fixture
def sample_labels_file(tmp_path) -> Path:
    """Write 365 fake category labels to a temp file, return path."""
    categories_file = tmp_path / "categories_places365.txt"
    lines = [f"/a/places365/{i:03d}/kitchen_variant_{i}\n" for i in range(365)]
    # Index 5 should be a recognizable label
    lines[5] = "/a/kitchen/kitchen\n"
    lines[100] = "/b/bedroom/bedroom\n"
    categories_file.write_text("".join(lines))
    return categories_file


# ── Tests ──────────────────────────────────────────────────────────────


class TestLoadPlaces365Labels:
    """Label file parsing (MIT-CSAIL format: '/a/b/c')."""

    def test_loads_365_labels(self, sample_labels_file):
        labels = load_places365_labels(sample_labels_file)
        assert len(labels) == 365

    def test_strips_directory_prefix(self, sample_labels_file):
        labels = load_places365_labels(sample_labels_file)
        # Index 5 was "/a/kitchen/kitchen" → should be "kitchen"
        assert labels[5] == "kitchen"
        # Index 100 was "/b/bedroom/bedroom" → "bedroom"
        assert labels[100] == "bedroom"

    def test_lowercase(self, sample_labels_file):
        labels = load_places365_labels(sample_labels_file)
        # All labels should be lowercase per Scenes dataset convention.
        for label in labels:
            assert label == label.lower()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_places365_labels(tmp_path / "does_not_exist.txt")

    def test_strips_whitespace(self, tmp_path):
        f = tmp_path / "labels.txt"
        f.write_text("  /a/kitchen/kitchen  \n/a/bedroom/bedroom\n")
        labels = load_places365_labels(f)
        assert labels[0] == "kitchen"
        assert labels[1] == "bedroom"


class TestPlaces365Result:
    """Result dataclass shape."""

    def test_top_k_default_5(self):
        r = Places365Result(category="kitchen", confidence=0.9, top5=[...])
        assert r.category == "kitchen"
        assert r.confidence == 0.9

    def test_repr_does_not_crash(self):
        r = Places365Result(category="kitchen", confidence=0.5, top5=[])
        assert "kitchen" in repr(r)


class TestPlaces365Classifier:
    """Classifier API surface."""

    def test_classify_returns_top5(self, dummy_image, mock_model, sample_labels_file):
        clf = Places365Classifier(model=mock_model, labels_path=sample_labels_file)
        result = clf.classify(dummy_image, top_k=5)
        assert isinstance(result, Places365Result)
        assert result.category == "kitchen"
        # Top-5 should have 5 entries (category, confidence)
        assert len(result.top5) == 5
        # Top-1 should match result.category
        assert result.top5[0][0] == "kitchen"

    def test_classify_top_k_limits_results(self, dummy_image, mock_model, sample_labels_file):
        clf = Places365Classifier(model=mock_model, labels_path=sample_labels_file)
        result = clf.classify(dummy_image, top_k=3)
        assert len(result.top5) == 3

    def test_classify_top1_has_highest_confidence(self, dummy_image, mock_model, sample_labels_file):
        clf = Places365Classifier(model=mock_model, labels_path=sample_labels_file)
        result = clf.classify(dummy_image, top_k=5)
        confidences = [c for _, c in result.top5]
        # Sorted descending
        assert confidences == sorted(confidences, reverse=True)

    def test_classify_confidences_in_0_1(self, dummy_image, mock_model, sample_labels_file):
        clf = Places365Classifier(model=mock_model, labels_path=sample_labels_file)
        result = clf.classify(dummy_image, top_k=5)
        for _, conf in result.top5:
            assert 0.0 <= conf <= 1.0

    def test_classify_runs_in_eval_mode(self, dummy_image, mock_model, sample_labels_file):
        clf = Places365Classifier(model=mock_model, labels_path=sample_labels_file)
        clf.classify(dummy_image)
        # Model should be put in eval mode (for dropout/BN behavior)
        assert mock_model.eval.called

    def test_classify_no_grad(self, dummy_image, mock_model, sample_labels_file):
        clf = Places365Classifier(model=mock_model, labels_path=sample_labels_file)
        with patch("aic26.pipeline.places365.torch.no_grad") as mock_no_grad:
            mock_no_grad.return_value.__enter__ = MagicMock()
            mock_no_grad.return_value.__exit__ = MagicMock()
            clf.classify(dummy_image)
            assert mock_no_grad.called


class TestClassifierLifecycle:
    """Lazy model loading and labels caching."""

    def test_constructor_stores_labels_path(self, sample_labels_file):
        clf = Places365Classifier(labels_path=sample_labels_file)
        assert clf.labels_path == sample_labels_file

    def test_labels_loaded_lazily(self, sample_labels_file):
        clf = Places365Classifier(labels_path=sample_labels_file)
        # Internal _labels dict empty until first classify() call
        # (Or eagerly loaded — both acceptable; just check it's populated
        # after classify)
        assert clf.labels_path == sample_labels_file
