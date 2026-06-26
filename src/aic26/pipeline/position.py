"""
Bounding-box position classifier for YOLO detections.

Maps a YOLO-format bbox [x_min, y_min, x_max, y_max] to a 3×3 grid
position label (top-left, top, top-right, left, center, right,
bottom-left, bottom, bottom-right). Mirrors lifeXplore's
`-o person|position:top-left` filter syntax (paper-knowledge §2).

Design notes
------------
- Grid boundaries are at 1/3 and 2/3 of image width/height. Center
  column/row spans [1/3, 2/3). Edge columns/rows are [0, 1/3) and
  [2/3, 1.0]. We use half-open intervals so every (x, y) maps to
  exactly one cell — no boundary ambiguity at x=0.333 or x=0.667.
- A bbox is rejected (returns None) when its area is below
  POSITION_GRID_THRESHOLD (5% of image area). Tiny bboxes are
  unreliable: a 10px-wide object on a 4K image produces a
  meaningless position label.
- Coordinates outside the image (negative or > img_w/h) are clamped
  to [0, 1] before classification so detection boxes that extend
  past the frame still produce a valid cell.

References
----------
- lifeXplore 2024 §3 (Tables 2-3): sub-filter `|position:<cell>`.
- docs/specs/metadata-extraction.md §Open Questions 3.
"""
from __future__ import annotations

from aic26.config.constants import POSITION_GRID_THRESHOLD

# Grid boundaries: [0, 1/3) → left/top, [1/3, 2/3) → center, [2/3, 1.0] → right/bottom.
# We use half-open intervals on the low side, closed on the high side, so a
# point exactly at x=2/3 belongs to the "right" column (matches the natural
# reading of "right edge").
_LEFT_BOUNDARY = 1.0 / 3.0
_RIGHT_BOUNDARY = 2.0 / 3.0


def classify_position(norm_x: float, norm_y: float) -> str:
    """Map normalized (x, y) coordinates ∈ [0, 1] to a 3×3 grid label.

    Args:
        norm_x: Horizontal center, 0 = left edge, 1 = right edge.
        norm_y: Vertical center, 0 = top edge, 1 = bottom edge.

    Returns:
        One of "top-left", "top", "top-right", "left", "center", "right",
        "bottom-left", "bottom", "bottom-right".
    """
    # Clamp to [0, 1] in case the caller passed slightly out-of-range values.
    nx = max(0.0, min(1.0, norm_x))
    ny = max(0.0, min(1.0, norm_y))

    if ny < _LEFT_BOUNDARY:
        row = "top"
    elif ny < _RIGHT_BOUNDARY:
        row = ""
    else:
        row = "bottom"

    if nx < _LEFT_BOUNDARY:
        col = "left"
    elif nx < _RIGHT_BOUNDARY:
        col = ""
    else:
        col = "right"

    if not row and not col:
        return "center"
    if row and col:
        return f"{row}-{col}"
    # Only row or only col (e.g. "top" alone, not "top-center")
    return row or col


def position_from_bbox(
    bbox: list[float],
    img_w: int,
    img_h: int,
    min_bbox_fraction: float = POSITION_GRID_THRESHOLD,
) -> str | None:
    """Map a YOLO xyxy bbox to a 3×3 grid label.

    Args:
        bbox: [x_min, y_min, x_max, y_max] in absolute pixels. Format matches
            ultralytics YOLO output (see pipeline/detection.py:276).
        img_w: Image width in pixels (must be > 0).
        img_h: Image height in pixels (must be > 0).
        min_bbox_fraction: Minimum bbox area as fraction of image area to
            emit a label. Default 0.05 (POSITION_GRID_THRESHOLD).

    Returns:
        Grid label string, or None if the bbox is degenerate (zero area,
        below the size threshold, or image dimensions are non-positive).
    """
    if img_w <= 0 or img_h <= 0:
        return None
    if len(bbox) != 4:
        return None

    x_min, y_min, x_max, y_max = bbox
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    if bbox_w <= 0 or bbox_h <= 0:
        return None

    bbox_area = bbox_w * bbox_h
    image_area = img_w * img_h
    if bbox_area / image_area < min_bbox_fraction:
        return None

    # Compute center of bbox and normalize to [0, 1].
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    norm_x = center_x / img_w
    norm_y = center_y / img_h

    return classify_position(norm_x, norm_y)
