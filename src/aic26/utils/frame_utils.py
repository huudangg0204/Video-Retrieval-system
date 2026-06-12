"""
Frame number parsing and CSV column detection utilities.

Handles extraction of frame numbers from filenames and
automatic detection of column names in mapping CSV files.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple
import pandas as pd


def parse_frame_number_from_filename(filename: str) -> Optional[int]:
    """
    Extract the trailing integer from a frame filename.
    
    Priority:
        1. Up to 6 trailing digits (e.g., "001" → 1, "000042" → 42)
        2. First digit sequence anywhere in the name as fallback
    
    Args:
        filename: Filename with or without extension
        
    Returns:
        Integer frame number, or None if no digits found
        
    Example:
        parse_frame_number_from_filename("frame_001.jpg")  # → 1
        parse_frame_number_from_filename("042.png")        # → 42
        parse_frame_number_from_filename("image.jpg")      # → None
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
    
    Searches for common column name patterns used in keyframe mapping CSVs.
    
    Args:
        df: Pandas DataFrame from a mapping CSV file
        
    Returns:
        Tuple of (frame_number_column, frame_index_column)
        
    Raises:
        KeyError: If required columns are not found
        
    Example:
        # CSV with columns: n, pst_time, fps, frame_index
        n_col, idx_col = detect_map_columns(df)  # → ("n", "frame_index")
    """
    candidates_n = ["n", "frame_number", "frame_no", "frame", "keyframe_id"]
    candidates_idx = ["frame_idx", "frame_index", "index", "idx", "video_frame"]
    
    n_col = _first_matching_column(df, candidates_n)
    idx_col = _first_matching_column(df, candidates_idx)
    
    if n_col is None or idx_col is None:
        raise KeyError(
            f"Mapping CSV missing expected columns. "
            f"Need one of {candidates_n} and one of {candidates_idx}. "
            f"Found columns: {list(df.columns)}"
        )
    
    return n_col, idx_col


def parse_frame_int(s) -> int:
    """
    Convert a frame identifier to an integer safely.
    
    Handles strings, integers, and filenames.
    Falls back to parse_frame_number_from_filename.
    Returns 0 on complete failure.
    
    Args:
        s: Frame identifier (int, str, or filename)
        
    Returns:
        Integer frame number (0 if parsing fails)
        
    Example:
        parse_frame_int(42)         # → 42
        parse_frame_int("042")      # → 42
        parse_frame_int("abc")      # → 0
    """
    try:
        return int(s)
    except (TypeError, ValueError):
        result = parse_frame_number_from_filename(str(s))
        return result if result is not None else 0


# ── Private helpers ─────────────────────────────────────────────────────

def _split_ext_safe(path: str) -> Tuple[str, str]:
    """Split filename and extension safely."""
    import os
    try:
        return os.path.splitext(str(path))
    except Exception:
        return str(path), ""


def _try_int(s: str) -> Optional[int]:
    """Try to convert string to int, return None on failure."""
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _first_matching_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find first column name that exists in the DataFrame."""
    for c in candidates:
        if c in df.columns:
            return c
    return None