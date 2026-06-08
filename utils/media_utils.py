"""utils/media_utils.py — Video mapping & timestamp helpers."""
import csv
import json
import os
from typing import Optional

import pandas as pd

from .frame_utils import detect_map_columns
from config import MAP_KEYFRAME_PATH, MEDIA_INFO_PATH


def load_fps_from_map(video_name: str, map_dir: str = MAP_KEYFRAME_PATH) -> Optional[float]:
    """
    Read the fps field from the first row of the keyframe-mapping CSV for *video_name*.

    Returns None if the file does not exist or the row is empty.
    """
    map_file = os.path.join(map_dir, f"{video_name}.csv")
    if not os.path.exists(map_file):
        return None
    try:
        with open(map_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return float(row["fps"])
    except Exception as e:
        print(f"⚠️ load_fps_from_map error for {video_name}: {e}")
    return None


def load_frame_idx_from_map(
    video_name: str,
    n: int,
    map_dir: str = MAP_KEYFRAME_PATH,
) -> Optional[int]:
    """
    Look up the frame_index for keyframe number *n* in the mapping CSV.

    Returns None if not found or on error.
    """
    map_file = os.path.join(map_dir, f"{video_name}.csv")
    if not os.path.exists(map_file):
        return None
    try:
        df = pd.read_csv(map_file)
        n_col, i_col = detect_map_columns(df)
        result = df.loc[df[n_col] == n, i_col]
        if not result.empty:
            return int(result.iloc[0])
    except Exception as e:
        print(f"⚠️ load_frame_idx_from_map error for {video_name} n={n}: {e}")
    return None


def frame_to_timestr(frame_idx: int, fps: float) -> str:
    """Convert a frame index to a human-readable MM:SS string."""
    if not fps or fps <= 0:
        return "0:00"
    sec = frame_idx / fps
    minutes = int(sec // 60)
    seconds = int(sec % 60)
    return f"{minutes}:{seconds:02d}"


def load_media_info(video_name: str, media_dir: str = MEDIA_INFO_PATH) -> dict:
    """
    Load the media-info JSON for *video_name*.

    Returns an empty dict if the file doesn't exist or is malformed.
    """
    media_file = os.path.join(media_dir, f"{video_name}.json")
    if not os.path.exists(media_file):
        return {}
    try:
        with open(media_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ load_media_info error for {video_name}: {e}")
        return {}
