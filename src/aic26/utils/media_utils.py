"""
Video media information utilities.

Handles reading FPS, frame indices, and timestamps from
keyframe mapping CSVs and media info JSON files.

Optimized: all CSV/JSON files are loaded once into module-level
caches for O(1) lookup during search.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

from loguru import logger

from aic26.utils.frame_utils import detect_map_columns


# ── Module-level caches (loaded once, shared across requests) ──

_fps_cache: Dict[str, Optional[float]] = {}
_frame_map_cache: Dict[str, Dict[int, int]] = {}  # video_name → {keyframe_n → frame_idx_video}
_media_info_cache: Dict[str, dict] = {}
_caches_initialized = False


def _init_caches() -> None:
    """Pre-load all map-keyframe CSVs and media-info JSONs into memory."""
    global _caches_initialized
    if _caches_initialized:
        return

    from aic26.config import settings

    map_dir = str(settings.map_keyframe_dir)
    media_dir = str(settings.media_info_dir)

    # Load all keyframe mapping CSVs
    if os.path.isdir(map_dir):
        for csv_file in sorted(os.listdir(map_dir)):
            if not csv_file.endswith(".csv"):
                continue
            video_name = os.path.splitext(csv_file)[0]
            csv_path = os.path.join(map_dir, csv_file)
            try:
                df = pd.read_csv(csv_path)
                n_col, idx_col = detect_map_columns(df)

                # Cache FPS from first row
                if "fps" in df.columns and len(df) > 0:
                    _fps_cache[video_name] = float(df["fps"].iloc[0])
                else:
                    _fps_cache[video_name] = None

                # Cache frame mapping: keyframe_n → frame_idx_video
                mapping = {}
                for _, row in df.iterrows():
                    mapping[int(row[n_col])] = int(row[idx_col])
                _frame_map_cache[video_name] = mapping

            except Exception as e:
                logger.warning(f"Failed to load map CSV {csv_file}: {e}")
                _fps_cache[video_name] = None
                _frame_map_cache[video_name] = {}

    # Load all media info JSONs
    if os.path.isdir(media_dir):
        for json_file in sorted(os.listdir(media_dir)):
            if not json_file.endswith(".json"):
                continue
            video_name = os.path.splitext(json_file)[0]
            json_path = os.path.join(media_dir, json_file)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    _media_info_cache[video_name] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load media info {json_file}: {e}")
                _media_info_cache[video_name] = {}

    _caches_initialized = True
    logger.info(
        f"Media caches loaded: {len(_fps_cache)} FPS entries, "
        f"{len(_frame_map_cache)} frame maps, "
        f"{len(_media_info_cache)} media info entries"
    )


def load_fps_from_map(
    video_name: str,
    map_dir: Optional[str] = None,
) -> Optional[float]:
    """
    Read the FPS for a video (from pre-loaded cache).
    
    Args:
        video_name: Name of the video (without extension)
        map_dir: Ignored (kept for backward compatibility)
        
    Returns:
        FPS as float, or None if not found
    """
    _init_caches()
    return _fps_cache.get(video_name)


def load_frame_idx_from_map(
    video_name: str,
    n: int,
    map_dir: Optional[str] = None,
) -> Optional[int]:
    """
    Look up the video frame index for keyframe number n (from pre-loaded cache).
    
    Args:
        video_name: Name of the video
        n: Keyframe number (as stored in mapping CSV)
        map_dir: Ignored (kept for backward compatibility)
        
    Returns:
        Frame index in the original video, or None if not found
    """
    _init_caches()
    mapping = _frame_map_cache.get(video_name)
    if mapping is None:
        return None
    return mapping.get(n)


def frame_to_timestr(frame_idx: int, fps: float) -> str:
    """
    Convert a frame index to human-readable time string (MM:SS).
    
    Args:
        frame_idx: Frame index in the video
        fps: Frames per second
        
    Returns:
        Time string in MM:SS format
    """
    if not fps or fps <= 0:
        return "0:00"
    
    sec = frame_idx / fps
    minutes = int(sec // 60)
    seconds = int(sec % 60)
    
    return f"{minutes}:{seconds:02d}"


def load_media_info(
    video_name: str,
    media_dir: Optional[str] = None,
) -> dict:
    """
    Load the media info for a video (from pre-loaded cache).
    
    Args:
        video_name: Name of the video
        media_dir: Ignored (kept for backward compatibility)
        
    Returns:
        Dictionary with media info, empty dict if not found
    """
    _init_caches()
    return _media_info_cache.get(video_name, {})