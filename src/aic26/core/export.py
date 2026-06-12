"""
CSV export utilities for competition submissions.

Converts retrieval results into properly formatted submission files.

Migrated from: core/export.py
"""
from __future__ import annotations

import csv
import os
import time
from typing import Dict, List, Optional

import pandas as pd

from loguru import logger

from aic26.config import settings
from aic26.utils.file_utils import ensure_dir
from aic26.utils.frame_utils import (
    parse_frame_number_from_filename,
    detect_map_columns,
)


class Exporter:
    """
    Converts retrieval results to competition submission CSV files.
    
    Supports three export formats:
    - Standard: video_name, frame_index [, const_value]
    - TRAKE: video_name, frame_1, frame_2, ..., frame_N
    - Submission: 20-row centered on a single frame
    
    Usage:
        exporter = Exporter()
        path = exporter.export_results(results, "my_submission")
    """
    
    def export_results(
        self,
        results: List[Dict],
        output_name: str,
        const_value: Optional[int] = None,
    ) -> str:
        """
        Export search results to standard submission CSV.
        
        Format: video_name,frame_index[,const_value]
        
        Args:
            results: List of result dicts with 'image_path' key
            output_name: Name for the output file (without .csv)
            const_value: Optional constant value for all rows
            
        Returns:
            Path to the written CSV file
            
        Raises:
            ValueError: If results list is empty
        """
        if not results:
            raise ValueError("No results to export")
        
        ensure_dir(str(settings.output_dir))
        out_path = os.path.join(str(settings.output_dir), f"{output_name}.csv")
        
        rows_written = 0
        rows_skipped = 0
        
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            for r in results:
                img_path = r["image_path"]
                vid = os.path.basename(os.path.dirname(img_path))
                n = parse_frame_number_from_filename(os.path.basename(img_path))
                
                if n is None:
                    logger.warning(f"Skip: cannot parse frame number from {img_path}")
                    rows_skipped += 1
                    continue
                
                frame_index = self._lookup_frame_index(vid, n)
                if frame_index is None:
                    rows_skipped += 1
                    continue
                
                if const_value is None:
                    f.write(f"{vid},{frame_index}\n")
                else:
                    f.write(f"{vid},{frame_index},{const_value}\n")
                rows_written += 1
        
        logger.info(
            f"Exported {rows_written} rows (skipped {rows_skipped}) → {out_path}"
        )
        return out_path
    
    def export_trake(
        self,
        trake_payload: List[Dict],
        output_name: str,
    ) -> str:
        """
        Export TRAKE results to submission CSV.
        
        Format: video_name,frame_id_1,frame_id_2,...,frame_id_N
        Rows are grouped by event size.
        
        Args:
            trake_payload: List of dicts with:
                - 'video_name': str
                - 'frame_idx': keyframe number
                - 'event': total events per answer
            output_name: Name for the output file
            
        Returns:
            Path to the written CSV file
            
        Raises:
            ValueError: If payload is empty
        """
        if not trake_payload:
            raise ValueError("No TRAKE results to export")
        
        ensure_dir(str(settings.output_dir))
        out_path = os.path.join(str(settings.output_dir), f"{output_name}.csv")
        
        events_per_answer = -1
        buffer: List[tuple] = []
        rows_written = 0
        
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            for r in trake_payload:
                vid = r.get("video_name")
                kf = r.get("frame_idx")
                size = r.get("event", 3)
                
                if events_per_answer == -1:
                    events_per_answer = size
                
                if not vid or kf is None:
                    logger.warning(f"Skip invalid TRAKE entry: {r}")
                    continue
                
                frame_index = self._lookup_frame_index(vid, int(kf))
                if frame_index is None:
                    continue
                
                buffer.append((vid, frame_index))
                
                if len(buffer) == events_per_answer:
                    vid0 = buffer[0][0]
                    frames = ",".join(str(fid) for _, fid in buffer)
                    f.write(f"{vid0},{frames}\n")
                    buffer.clear()
                    rows_written += 1
        
        logger.info(f"Exported {rows_written} TRAKE rows → {out_path}")
        return out_path
    
    def export_submission(
        self,
        video_name: str,
        frame_id: int,
        gap: int,
        const_value: Optional[int] = None,
        name: Optional[str] = None,
    ) -> str:
        """
        Generate a 20-row submission CSV centered on a single frame.
        
        Creates rows at: frame_id, frame_id ± gap, frame_id ± 2*gap, ...
        
        Args:
            video_name: Name of the video
            frame_id: Center frame index
            gap: Gap between adjacent frames
            const_value: Optional constant value
            name: Output file name (auto-generated if None)
            
        Returns:
            Path to the written CSV file
            
        Raises:
            ValueError: If required parameters are missing
        """
        if video_name is None or frame_id is None or gap is None:
            raise ValueError("video_name, frame_id, gap are all required")
        
        frame_id = int(frame_id)
        gap = int(gap)
        
        if gap <= 0:
            raise ValueError("gap must be > 0")
        
        if name is None:
            name = f"submission_{int(time.time())}"
        
        ensure_dir(str(settings.output_dir))
        csv_path = os.path.join(str(settings.output_dir), f"{name}.csv")
        
        rows: List[List] = []
        offset = 0
        
        while len(rows) < 20:
            if const_value is not None:
                center_row = [video_name, frame_id + offset, const_value]
            else:
                center_row = [video_name, frame_id + offset]
            rows.append(center_row)
            
            if offset > 0:
                minus_id = frame_id - offset
                if minus_id >= 0:
                    if const_value is not None:
                        rows.append([video_name, minus_id, const_value])
                    else:
                        rows.append([video_name, minus_id])
            
            offset += gap
        
        rows = rows[:20]
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        logger.info(f"Exported submission ({len(rows)} rows) → {csv_path}")
        return csv_path
    
    # ── Private ───────────────────────────────────────────────────
    
    @staticmethod
    def _lookup_frame_index(vid: str, n: int) -> Optional[int]:
        """
        Map keyframe number to video frame index via mapping CSV.
        
        Args:
            vid: Video name
            n: Keyframe number
            
        Returns:
            Frame index in the original video, or None
        """
        map_csv = os.path.join(str(settings.map_keyframe_dir), f"{vid}.csv")
        
        if not os.path.exists(map_csv):
            logger.warning(f"Mapping CSV not found: {map_csv}")
            return None
        
        try:
            df = pd.read_csv(map_csv)
            n_col, idx_col = detect_map_columns(df)
            sub = df.loc[df[n_col] == n, idx_col]
            
            if sub.empty:
                logger.warning(f"Frame {n} not in mapping for {vid}")
                return None
            
            return int(sub.iloc[0])
            
        except Exception as e:
            logger.warning(f"Mapping read error for {vid}: {e}")
            return None