"""core/export.py — Submission CSV export helpers.

Fixes vs. original:
1. export_trake was defined TWICE (lines 919 + 946 in original); the first incomplete
   version shadowed the second. Only ONE correct implementation exists here.
2. export_submission's `import time` is now at the top of the file.
3. All path handling uses config constants.
4. Errors are raised explicitly instead of silently continuing.
"""
from __future__ import annotations

import csv
import os
import time
from typing import Dict, List, Optional

import pandas as pd

import config
from utils.file_utils import ensure_dir
from utils.frame_utils import parse_frame_number_from_filename, detect_map_columns


class Exporter:
    """Converts retrieval results into competition submission CSV files."""

    def export_results(
        self,
        results: List[Dict],
        output_name: str,
        const_value: Optional[int] = None,
    ) -> str:
        """
        Write a standard submission CSV (video_name, frame_index[, const_value]).

        Each result dict must contain 'image_path'.
        Returns the path to the written CSV.
        """
        if not results:
            raise ValueError("No results to export")

        ensure_dir(config.OUTPUT_FOLDER)
        out_path = os.path.join(config.OUTPUT_FOLDER, f"{output_name}.csv")

        with open(out_path, "w", encoding="utf-8", newline="") as f:
            for r in results:
                img_path = r["image_path"]
                vid      = os.path.basename(os.path.dirname(img_path))
                n        = parse_frame_number_from_filename(os.path.basename(img_path))

                if n is None:
                    print(f"⚠️ Skip: cannot parse frame number from {img_path}")
                    continue

                frame_index = self._lookup_frame_index(vid, n)
                if frame_index is None:
                    continue

                if const_value is None:
                    f.write(f"{vid},{frame_index}\n")
                else:
                    f.write(f"{vid},{frame_index},{const_value}\n")

        print(f"📄 Exported submission → {out_path}")
        return out_path

    def export_trake(
        self,
        trake_payload: List[Dict],
        output_name: str,
    ) -> str:
        """
        Write a TRAKE submission CSV.

        Each entry in *trake_payload* is a flat result dict with:
            - 'video_name' : str
            - 'frame_idx'  : keyframe number (str or int)
            - 'event'      : total number of events per answer (int)

        Rows are accumulated in groups of *event* size and written as:
            video_name, frame_id_1, frame_id_2, ..., frame_id_N
        """
        if not trake_payload:
            raise ValueError("No TRAKE results to export")

        ensure_dir(config.OUTPUT_FOLDER)
        out_path = os.path.join(config.OUTPUT_FOLDER, f"{output_name}.csv")

        events_per_answer = -1
        buffer: List[tuple] = []

        with open(out_path, "w", encoding="utf-8", newline="") as f:
            for r in trake_payload:
                vid  = r.get("video_name")
                kf   = r.get("frame_idx")
                size = r.get("event", 3)

                if events_per_answer == -1:
                    events_per_answer = size

                if not vid or kf is None:
                    print(f"⚠️ Skip invalid entry: {r}")
                    continue

                frame_index = self._lookup_frame_index(vid, int(kf))
                if frame_index is None:
                    continue

                buffer.append((vid, frame_index))

                if len(buffer) == events_per_answer:
                    vid0   = buffer[0][0]
                    frames = ",".join(str(fid) for _, fid in buffer)
                    f.write(f"{vid0},{frames}\n")
                    buffer.clear()

        print(f"📄 Exported TRAKE submission → {out_path}")
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
        Generate a 20-row submission CSV centred on *frame_id* with ±gap offsets.

        Useful for single-frame QA where you want to submit nearby frames as
        alternative answers.
        """
        if video_name is None or frame_id is None or gap is None:
            raise ValueError("video_name, frame_id, gap are all required")

        frame_id = int(frame_id)
        gap      = int(gap)
        if gap <= 0:
            raise ValueError("gap must be > 0")

        if name is None:
            name = f"submission_{int(time.time())}"

        ensure_dir(config.OUTPUT_FOLDER)
        csv_path = os.path.join(config.OUTPUT_FOLDER, f"{name}.csv")

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

        return csv_path

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _lookup_frame_index(vid: str, n: int) -> Optional[int]:
        """Map keyframe number *n* → video frame index via the mapping CSV."""
        map_csv = os.path.join(config.MAP_KEYFRAME_PATH, f"{vid}.csv")
        if not os.path.exists(map_csv):
            print(f"⚠️ Mapping CSV not found: {map_csv}")
            return None
        try:
            df       = pd.read_csv(map_csv)
            n_col, idx_col = detect_map_columns(df)
            sub      = df.loc[df[n_col] == n, idx_col]
            if sub.empty:
                print(f"⚠️ Frame {n} not in mapping for {vid}")
                return None
            return int(sub.iloc[0])
        except Exception as e:
            print(f"⚠️ Mapping read error for {vid}: {e}")
            return None
