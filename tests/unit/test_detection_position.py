"""
Unit tests for the position enrichment added to ObjectDetector.

The pure helper `add_position_to_detections(detections, img_h, img_w)`
takes the existing YOLO detection dict and appends a new
`detection_positions` list. Existing fields are not modified, so
downstream consumers that rely on the old shape keep working.

Detection dict shape (from ObjectDetector._detect_single):
    {
        "detection_scores": list[float],
        "detection_class_names": list[str],   # MID
        "detection_class_entities": list[str],
        "detection_boxes": list[[x1, y1, x2, y2]],
        "detection_class_labels": list[int],
    }
"""
from __future__ import annotations

from aic26.pipeline.detection import add_position_to_detections


def _det(bbox: list[float], label: str = "person", score: float = 0.9) -> dict:
    """Build a minimal detection dict containing one bbox."""
    return {
        "detection_scores": [score],
        "detection_class_names": [label],
        "detection_class_entities": [label.title()],
        "detection_boxes": [bbox],
        "detection_class_labels": [0],
    }


class TestAddPositionToDetections:
    """Pure helper that annotates the dict with detection_positions."""

    def test_adds_detection_positions_key(self):
        det = _det([10, 10, 160, 160])
        out = add_position_to_detections(det, img_h=360, img_w=640)
        assert "detection_positions" in out

    def test_position_label_correct_top_left(self):
        det = _det([10, 10, 160, 160])  # bbox 150x150 ≈ 9.77% of 640x360 → top-left
        out = add_position_to_detections(det, img_h=360, img_w=640)
        assert out["detection_positions"] == ["top-left"]

    def test_position_label_correct_center(self):
        det = _det([245, 110, 395, 250])
        out = add_position_to_detections(det, img_h=360, img_w=640)
        assert out["detection_positions"] == ["center"]

    def test_position_label_correct_bottom_right(self):
        det = _det([480, 200, 630, 350])
        out = add_position_to_detections(det, img_h=360, img_w=640)
        assert out["detection_positions"] == ["bottom-right"]

    def test_tiny_bbox_position_is_none(self):
        # 10x10 = 100 / 230400 = 0.043% < 5% threshold → None
        det = _det([100, 100, 110, 110])
        out = add_position_to_detections(det, img_h=360, img_w=640)
        assert out["detection_positions"] == [None]

    def test_zero_area_bbox_position_is_none(self):
        det = _det([100, 100, 100, 100])
        out = add_position_to_detections(det, img_h=360, img_w=640)
        assert out["detection_positions"] == [None]

    def test_multiple_bboxes_all_get_positions(self):
        det = {
            "detection_scores": [0.9, 0.8, 0.7],
            "detection_class_names": ["person", "car", "cup"],
            "detection_class_entities": ["Person", "Car", "Cup"],
            "detection_boxes": [
                [10, 10, 160, 160],   # top-left
                [245, 110, 395, 250],  # center
                [480, 200, 630, 350],  # bottom-right
            ],
            "detection_class_labels": [0, 1, 2],
        }
        out = add_position_to_detections(det, img_h=360, img_w=640)
        assert out["detection_positions"] == ["top-left", "center", "bottom-right"]

    def test_existing_fields_unchanged(self):
        det = _det([10, 10, 160, 160])
        out = add_position_to_detections(det, img_h=360, img_w=640)
        # Original keys/values preserved exactly.
        assert out["detection_scores"] == det["detection_scores"]
        assert out["detection_class_names"] == det["detection_class_names"]
        assert out["detection_class_entities"] == det["detection_class_entities"]
        assert out["detection_boxes"] == det["detection_boxes"]
        assert out["detection_class_labels"] == det["detection_class_labels"]

    def test_returns_new_dict_does_not_mutate_input(self):
        det = _det([10, 10, 160, 160])
        add_position_to_detections(det, img_h=360, img_w=640)
        assert "detection_positions" not in det  # input untouched

    def test_empty_detections_returns_empty_list(self):
        det = {
            "detection_scores": [],
            "detection_class_names": [],
            "detection_class_entities": [],
            "detection_boxes": [],
            "detection_class_labels": [],
        }
        out = add_position_to_detections(det, img_h=360, img_w=640)
        assert out["detection_positions"] == []

    def test_mismatched_lengths_raises(self):
        # Defensive: a corrupt detection dict should fail loudly, not silently.
        det = {
            "detection_scores": [0.9, 0.8],  # 2 scores
            "detection_class_names": ["person"],  # 1 name — inconsistent
            "detection_class_entities": ["Person"],
            "detection_boxes": [[10, 10, 160, 160]],
            "detection_class_labels": [0],
        }
        import pytest
        with pytest.raises(ValueError):
            add_position_to_detections(det, img_h=360, img_w=640)

    def test_custom_threshold(self):
        # 150x150 bbox = 9.77% → above 0.05 threshold → top-left
        # With threshold 0.10 it becomes None (below 10%)
        det = _det([10, 10, 160, 160])
        out = add_position_to_detections(
            det, img_h=360, img_w=640, min_bbox_fraction=0.10,
        )
        assert out["detection_positions"] == [None]
