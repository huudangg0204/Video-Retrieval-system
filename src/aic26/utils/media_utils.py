"""
Video media information utilities.

Handles reading FPS, frame indices, and timestamps from
keyframe mapping CSVs and media info JSON files.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Optional
import pandas as pd

from aic26.utils.frame_utils import detect_map_columns


def load_fps_from_map(
    video_name: str,
    map_dir: Optional[str] = None,
) -> Optional[float]:
    """
    Read the FPS from a keyframe mapping CSV file.
    
    Args:
        video_name: Name of the video (without extension)
        map_dir: Directory containing mapping CSV files
                (defaults to settings.map_keyframe_dir)
        
    Returns:
        FPS as float, or None if not found
        
    Example:
        fps = load_fps_from_map("L21_V001")  # → 30.0
    """
    from aic26.config import settings
    
    map_dir = map_dir or str(settings.map_keyframe_dir)
    map_file = os.path.join(map_dir, f"{video_name}.csv")
    
    if not os.path.exists(map_file):
        return None
    
    try:
        with open(map_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "fps" in row:
                    return float(row["fps"])
                break  # Only need first row
    except Exception as e:
        from loguru import logger
        logger.warning(f"Could not load FPS for {video_name}: {e}")
    
    return None


def load_frame_idx_from_map(
    video_name: str,
    n: int,
    map_dir: Optional[str] = None,
) -> Optional[int]:
    """
    Look up the video frame index for keyframe number n.
    
    Args:
        video_name: Name of the video
        n: Keyframe number (as stored in mapping CSV)
        map_dir: Directory containing mapping CSV files
        
    Returns:
        Frame index in the original video, or None if not found
        
    Example:
        # Keyframe #3 corresponds to frame 96 in the video
        idx = load_frame_idx_from_map("L21_V001", 3)  # → 96
    """
    from aic26.config import settings
    
    map_dir = map_dir or str(settings.map_keyframe_dir)
    map_file = os.path.join(map_dir, f"{video_name}.csv")
    
    if not os.path.exists(map_file):
        return None
    
    try:
        df = pd.read_csv(map_file)
        n_col, idx_col = detect_map_columns(df)
        result = df.loc[df[n_col] == n, idx_col]
        if not result.empty:
            return int(result.iloc[0])
    except Exception as e:
        from loguru import logger
        logger.warning(f"Could not load frame index for {video_name} n={n}: {e}")
    
    return None


def frame_to_timestr(frame_idx: int, fps: float) -> str:
    """
    Convert a frame index to human-readable time string (MM:SS).
    
    Args:
        frame_idx: Frame index in the video
        fps: Frames per second
        
    Returns:
        Time string in MM:SS format
        
    Example:
        frame_to_timestr(90, 30)   # → "0:03"
        frame_to_timestr(3600, 30) # → "2:00"
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
    Load the media info JSON for a video.
    
    Args:
        video_name: Name of the video
        media_dir: Directory containing media info JSON files
        
    Returns:
        Dictionary with media info, empty dict if not found
        
    Example:
        info = load_media_info("L21_V001")
        # → {"duration": 3600, "watch_url": "https://...", ...}
    """
    from aic26.config import settings
    
    media_dir = media_dir or str(settings.media_info_dir)
    media_file = os.path.join(media_dir, f"{video_name}.json")
    
    if not os.path.exists(media_file):
        return {}
    
    try:
        with open(media_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        from loguru import logger
        logger.warning(f"Could not load media info for {video_name}: {e}")
        return {}