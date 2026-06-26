"""
Unit tests for pipeline/position.py.

Tests the bbox → 3×3 grid position classifier used to enrich YOLO
detections with spatial labels (per docs/specs/metadata-extraction.md).

Bbox format: [x_min, y_min, x_max, y_max] in absolute pixel coordinates
(matches ultralytics YOLO output, see pipeline/detection.py:276).
"""
from __future__ import annotations

import pytest

from aic26.pipeline.position import position_from_bbox, classify_position
from aic26.config.constants import (
    POSITION_GRID_3x3,
    POSITION_GRID_THRESHOLD,
)


# Image 640x360 — wide-screen keyframe.
IMG_W, IMG_H = 640, 360


class TestPositionFromBbox:
    """Core function: bbox → grid label."""

    def test_top_left_corner(self):
        # bbox 150×150 = 22500 px² ≈ 9.77% of 640×360 — above 5% threshold
        assert position_from_bbox([10, 10, 160, 160], IMG_W, IMG_H) == "top-left"

    def test_top_center(self):
        # bbox 150×150 — center column, top row
        assert position_from_bbox([245, 10, 395, 160], IMG_W, IMG_H) == "top"

    def test_top_right_corner(self):
        assert position_from_bbox([480, 10, 630, 160], IMG_W, IMG_H) == "top-right"

    def test_middle_left(self):
        assert position_from_bbox([10, 110, 160, 250], IMG_W, IMG_H) == "left"

    def test_center(self):
        # bbox 150×150 — center column, center row
        assert position_from_bbox([245, 110, 395, 250], IMG_W, IMG_H) == "center"

    def test_middle_right(self):
        assert position_from_bbox([480, 110, 630, 250], IMG_W, IMG_H) == "right"

    def test_bottom_left_corner(self):
        assert position_from_bbox([10, 200, 160, 350], IMG_W, IMG_H) == "bottom-left"

    def test_bottom_center(self):
        assert position_from_bbox([245, 200, 395, 350], IMG_W, IMG_H) == "bottom"

    def test_bottom_right_corner(self):
        assert position_from_bbox([480, 200, 630, 350], IMG_W, IMG_H) == "bottom-right"


class TestPositionThresholds:
    """Bbox too small → None; bbox boundary cases."""

    def test_tiny_bbox_returns_none(self):
        # bbox width=2, height=2 → area = 4/230400 = ~0.0017 < 0.05
        bbox = [100, 100, 102, 102]
        assert position_from_bbox(bbox, IMG_W, IMG_H) is None

    def test_bbox_exactly_at_threshold_returns_label(self):
        # 5% of 640*360 = 11520 → bbox area must be ≥ that.
        # bbox [0,0,11520^0.5,11520^0.5] = [0,0,107.3,107.3]
        # 107.3^2 = 11513 ≈ threshold.
        # Use slightly larger bbox to be safely above threshold.
        bbox = [0, 0, 110, 110]  # area = 12100, ratio = 0.0525
        result = position_from_bbox(bbox, IMG_W, IMG_H)
        assert result == "top-left"

    def test_bbox_just_below_threshold_returns_none(self):
        # area ratio 0.04 → below 0.05
        # 0.04 * 230400 = 9216 → sqrt = 96
        bbox = [0, 0, 96, 96]
        assert position_from_bbox(bbox, IMG_W, IMG_H) is None


class TestPositionEdgeCases:
    """Edge cases: zero-size bbox, bbox outside image, malformed."""

    def test_zero_area_bbox_returns_none(self):
        assert position_from_bbox([100, 100, 100, 100], IMG_W, IMG_H) is None

    def test_negative_bbox_coordinates_clamped(self):
        # bbox partly outside image; position is computed from center.
        # bbox [-50, -50, 250, 250] → center = (100, 100) → top-left
        # area = 300×300 = 90000, ratio = 90000/230400 = 39% — above threshold
        assert position_from_bbox([-50, -50, 250, 250], IMG_W, IMG_H) == "top-left"

    def test_full_image_bbox_returns_center(self):
        # bbox covers entire image → center
        result = position_from_bbox([0, 0, IMG_W, IMG_H], IMG_W, IMG_H)
        assert result == "center"

    def test_returns_string_in_grid(self):
        # bbox 150×150 = 22500 px² ≈ 9.77% of image — above threshold
        result = position_from_bbox([100, 100, 250, 250], IMG_W, IMG_H)
        assert result in POSITION_GRID_3x3


class TestClassifyPosition:
    """Functional API on normalized (0..1) coordinates — separate from bbox variant."""

    def test_normalized_center(self):
        assert classify_position(0.5, 0.5) == "center"

    def test_normalized_top_left(self):
        assert classify_position(0.1, 0.1) == "top-left"

    def test_normalized_bottom_right(self):
        assert classify_position(0.9, 0.9) == "bottom-right"

    def test_normalized_boundary_top_center(self):
        # x=0.5 exactly on boundary, y=0.1 → "top"
        assert classify_position(0.5, 0.1) == "top"

    def test_normalized_boundary_left_center(self):
        # x=0.1, y=0.5 exactly → "left"
        assert classify_position(0.1, 0.5) == "left"

    def test_returns_none_for_zero_dimensions(self):
        # Defensive: caller passes 0,0 should not crash.
        # Implementation should still classify based on coords (both < 1/3).
        assert classify_position(0.0, 0.0) == "top-left"
