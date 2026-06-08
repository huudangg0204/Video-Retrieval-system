"""utils/frame_utils.py — Frame number & CSV-column parsing helpers."""
import re
import pandas as pd
from typing import Optional, Tuple


def parse_frame_number_from_filename(filename: str) -> Optional[int]:
    """
    Extract the trailing integer from a frame filename (without extension).

    Priority:
      1. Up to 6 trailing digits  (e.g. "001", "000042")
      2. First digit sequence anywhere in the name as fallback

    Returns None if no digit sequence is found.
    """
    name, _ = _split_ext_safe(filename)
    # Primary: trailing digits (1–6 chars)
    m = re.search(r'(\d{1,6})$', name)
    if m:
        return _try_int(m.group(1))
    # Fallback: first digit sequence in the name
    m2 = re.search(r'(\d+)', name)
    if m2:
        return _try_int(m2.group(1))
    return None


def detect_map_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Detect the (frame_number_col, frame_index_col) pair in a mapping DataFrame.

    Raises KeyError with a descriptive message if columns are not found.
    """
    candidates_n   = ["n", "frame_number", "frame_no", "frame"]
    candidates_idx = ["frame_idx", "frame_index", "index", "idx"]

    n_col   = _first_matching_column(df, candidates_n)
    idx_col = _first_matching_column(df, candidates_idx)

    if n_col is None or idx_col is None:
        raise KeyError(
            f"Mapping CSV missing expected columns. "
            f"Need one of {candidates_n} and one of {candidates_idx}. "
            f"Found: {list(df.columns)}"
        )
    return n_col, idx_col


def parse_frame_int(s) -> int:
    """
    Convert a frame identifier (string or int) to an integer.
    Falls back to parse_frame_number_from_filename; returns 0 on failure.
    """
    try:
        return int(s)
    except (TypeError, ValueError):
        result = parse_frame_number_from_filename(str(s))
        return result if result is not None else 0


# ── private helpers ───────────────────────────────────────────────────────────

def _split_ext_safe(path: str) -> Tuple[str, str]:
    """os.path.splitext that never raises."""
    import os
    try:
        return os.path.splitext(path)
    except Exception:
        return str(path), ""


def _try_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _first_matching_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None
