"""
Unit tests for metadata extraction constants and settings.

Tests verify that constants added in Task 1 (constants + Settings)
are exposed correctly and Settings picks up the new env vars.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from aic26.config.constants import (
    POSITION_GRID_3x3,
    POSITION_GRID_THRESHOLD,
    SCENE_GRAPH_PROMPT,
    PLACES365_DEFAULT_MODEL,
)
from aic26.config.settings import Settings


class TestPositionConstants:
    """POSITION_GRID_3x3 + threshold for bbox filtering."""

    def test_position_grid_has_9_cells(self):
        assert len(POSITION_GRID_3x3) == 9

    def test_position_grid_expected_labels(self):
        expected = {
            "top-left", "top", "top-right",
            "left", "center", "right",
            "bottom-left", "bottom", "bottom-right",
        }
        assert set(POSITION_GRID_3x3) == expected

    def test_position_grid_is_tuple(self):
        # Constants are Final tuples per codebase convention
        assert isinstance(POSITION_GRID_3x3, tuple)

    def test_position_grid_threshold_is_fraction(self):
        assert 0.0 < POSITION_GRID_THRESHOLD < 0.5
        # Spec: skip position for bbox < 5% image size
        assert POSITION_GRID_THRESHOLD == pytest.approx(0.05, abs=1e-9)


class TestSceneGraphPrompt:
    """SCENE_GRAPH_PROMPT template constant."""

    def test_prompt_is_non_empty_string(self):
        assert isinstance(SCENE_GRAPH_PROMPT, str)
        assert len(SCENE_GRAPH_PROMPT.strip()) > 50

    def test_prompt_mentions_format(self):
        assert "JSON" in SCENE_GRAPH_PROMPT
        assert "subject" in SCENE_GRAPH_PROMPT.lower()
        assert "predicate" in SCENE_GRAPH_PROMPT.lower()
        assert "object" in SCENE_GRAPH_PROMPT.lower()

    def test_prompt_includes_rules(self):
        # SnapSeek §3.5: lowercase noun, base-form predicates, no passive
        assert "lowercase" in SCENE_GRAPH_PROMPT.lower()
        assert "base-form" in SCENE_GRAPH_PROMPT.lower() or "base form" in SCENE_GRAPH_PROMPT.lower()
        assert "passive" in SCENE_GRAPH_PROMPT.lower()


class TestPlaces365Default:
    def test_default_model_string(self):
        # Spec §AD-2: EfficientNet-B2
        assert "efficientnet" in PLACES365_DEFAULT_MODEL.lower() or "places365" in PLACES365_DEFAULT_MODEL.lower()


class TestMetadataSettings:
    """Settings env vars for metadata extraction."""

    def test_metadata_dir_default(self):
        s = Settings()
        assert s.metadata_dir == Path("data/metadata")

    def test_scene_graph_model_default(self):
        s = Settings()
        assert "Qwen2.5-VL" in s.scene_graph_model

    def test_places365_model_name_default(self):
        s = Settings()
        assert isinstance(s.places365_model_name, str)
        assert len(s.places365_model_name) > 0

    def test_scene_graph_max_triples_default(self):
        s = Settings()
        # SnapSeek §3.5: "Limit to the top 5 most salient triples"
        assert s.scene_graph_max_triples == 5

    def test_scene_graph_max_new_tokens_default(self):
        s = Settings()
        assert 128 <= s.scene_graph_max_new_tokens <= 2048

    def test_position_min_bbox_fraction_default(self):
        s = Settings()
        assert s.position_min_bbox_fraction == pytest.approx(0.05, abs=1e-9)

    def test_metadata_dir_in_ensure_directories(self, temp_dir):
        data_dir = temp_dir / "data"
        s = Settings(data_dir=data_dir)
        s.ensure_directories()
        assert (data_dir / "metadata").exists()
