"""
OCR Search Engine for AIC-26.

Loads pre-computed OCR results (CSV) và cung cấp full-text search
trên nội dung văn bản đã trích xuất từ keyframes.

Features:
- Substring search (case-insensitive)
- Regex search
- Combined: OCR text search → intersect với embedding search (re-rank)
- Lazy-load + auto-reload khi CSV thay đổi
- Enrich kết quả với timing metadata (frame_idx_video, time_str)
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from loguru import logger

from aic26.config import settings
from aic26.utils.media_utils import (
    frame_to_timestr,
    load_fps_from_map,
    load_frame_idx_from_map,
    load_media_info,
)
from aic26.utils.frame_utils import parse_frame_number_from_filename


def _normalize_text(text: str) -> str:
    """Chuẩn hóa text để so sánh: lowercase + bỏ dấu tiếng Việt."""
    # NFD decompose → remove combining diacritics → lowercase
    nfd = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return stripped.lower()


class OCRSearchEngine:
    """
    Full-text search trên kết quả OCR đã lưu trong CSV.

    Cách dùng:
        engine = OCRSearchEngine()
        results = engine.search("TRIỀU CƯỜNG", top_k=50)
    """

    def __init__(
        self,
        ocr_csv_path: Optional[str] = None,
    ) -> None:
        self._csv_path = Path(ocr_csv_path or str(settings.ocr_output_path))
        self._df: Optional[pd.DataFrame] = None
        self._mtime: float = 0.0

    # ── Public API ────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 100,
        mode: str = "substring",      # "substring" | "regex" | "exact"
        case_sensitive: bool = False,
        normalize_accents: bool = True,
        min_conf: float = 0.0,
        video_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm keyframes có chứa text khớp với query.

        Args:
            query: Chuỗi tìm kiếm (substring / regex pattern).
            top_k: Số kết quả tối đa trả về.
            mode: "substring" (chứa), "regex" (biểu thức chính quy), "exact" (khớp toàn bộ).
            case_sensitive: Phân biệt hoa/thường (mặc định không).
            normalize_accents: Bỏ dấu tiếng Việt khi so sánh (mặc định có).
            min_conf: Lọc theo confidence trung bình tối thiểu.
            video_filter: Nếu không rỗng, chỉ tìm trong video này.

        Returns:
            List of dicts: image_path, video_name, frame_idx,
                           ocr_text, num_boxes, conf_mean,
                           watch_url, fps, frame_idx_video, time_str.
        """
        df = self._load_df()
        if df is None or df.empty:
            logger.warning("OCR CSV rỗng hoặc không tồn tại.")
            return []

        if not query.strip():
            return []

        # ── Lọc theo video ─────────────────────────────────────────
        if video_filter:
            df = df[df["video"].astype(str).str.contains(video_filter, case=False, na=False)]

        # ── Lọc theo confidence ─────────────────────────────────────
        if min_conf > 0.0 and "conf_mean" in df.columns:
            df = df[pd.to_numeric(df["conf_mean"], errors="coerce").fillna(0.0) >= min_conf]

        # ── Chuẩn bị cột text để so sánh ───────────────────────────
        text_col = df["text_full"].fillna("").astype(str)

        if normalize_accents:
            compare_col = text_col.map(_normalize_text)
            compare_query = _normalize_text(query)
        else:
            compare_col = text_col if case_sensitive else text_col.str.lower()
            compare_query = query if case_sensitive else query.lower()

        # ── Match ───────────────────────────────────────────────────
        try:
            if mode == "exact":
                mask = compare_col == compare_query
            elif mode == "regex":
                flags = 0 if case_sensitive else re.IGNORECASE
                mask = compare_col.str.contains(compare_query, regex=True, flags=flags, na=False)
            else:  # substring (default)
                mask = compare_col.str.contains(re.escape(compare_query), regex=True, na=False)
        except re.error as e:
            logger.error(f"OCR search regex error: {e}")
            return []

        matched = df[mask]
        if matched.empty:
            return []

        # Giới hạn top_k
        matched = matched.head(top_k)

        # ── Build results ───────────────────────────────────────────
        results = []
        _media_cache: Dict[str, dict] = {}

        for _, row in matched.iterrows():
            video_name = str(row.get("video", ""))
            img_path = str(row.get("path", "")).replace(os.sep, "/")
            frame_str = os.path.splitext(str(row.get("file", "")))[0]

            if video_name not in _media_cache:
                _media_cache[video_name] = load_media_info(video_name)
            media_info = _media_cache[video_name]

            result: Dict[str, Any] = {
                "image_path": img_path,
                "video_name": video_name,
                "frame_idx": frame_str,
                "ocr_text": str(row.get("text_full", "")),
                "num_boxes": int(row.get("num_boxes", 0)),
                "conf_mean": float(row.get("conf_mean", 0.0)) if row.get("conf_mean") else None,
                "watch_url": media_info.get("watch_url", ""),
                "similarity": None,   # không có embedding score khi search OCR thuần
            }

            self._enrich_with_timing(result)
            results.append(result)

        logger.info(
            f"OCR search '{query[:60]}' → {len(results)} results "
            f"(mode={mode}, normalize={normalize_accents})"
        )
        return results

    def get_ocr_for_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin OCR của 1 ảnh cụ thể theo path.

        Args:
            image_path: Đường dẫn đầy đủ hoặc tương đối của ảnh.

        Returns:
            Dict OCR row hoặc None nếu không tìm thấy.
        """
        df = self._load_df()
        if df is None or df.empty:
            return None

        # Chuẩn hóa path separator
        norm_path = image_path.replace("\\", "/")
        match = df[df["path"].str.replace("\\", "/", regex=False) == norm_path]

        if match.empty:
            return None

        row = match.iloc[0]
        return {
            "image_path": norm_path,
            "video_name": str(row.get("video", "")),
            "file": str(row.get("file", "")),
            "text_full": str(row.get("text_full", "")),
            "num_boxes": int(row.get("num_boxes", 0)),
            "conf_mean": float(row.get("conf_mean", 0.0)) if row.get("conf_mean") else None,
            "boxes": self._parse_json_col(row.get("boxes")),
            "texts": self._parse_json_col(row.get("texts")),
            "confs": self._parse_json_col(row.get("confs")),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Trả về thống kê về OCR index."""
        df = self._load_df()
        if df is None or df.empty:
            return {"total_frames": 0, "total_videos": 0, "csv_exists": self._csv_path.exists()}

        return {
            "total_frames": len(df),
            "total_videos": df["video"].nunique() if "video" in df.columns else 0,
            "csv_path": str(self._csv_path),
            "csv_size_mb": round(self._csv_path.stat().st_size / 1024 / 1024, 2),
        }

    def reload(self) -> None:
        """Force reload CSV từ disk."""
        self._df = None
        self._mtime = 0.0
        self._load_df()

    # ── Private helpers ───────────────────────────────────────────────

    def _load_df(self) -> Optional[pd.DataFrame]:
        """Lazy-load CSV, auto-reload nếu file thay đổi."""
        if not self._csv_path.exists():
            return None

        mtime = self._csv_path.stat().st_mtime
        if self._df is not None and mtime == self._mtime:
            return self._df

        try:
            # CSV có thể không có header nếu bị ghi append không cẩn thận
            df = pd.read_csv(
                self._csv_path,
                header=None,
                names=["video", "file", "path", "text_full", "num_boxes", "conf_mean", "boxes", "texts", "confs"],
                encoding="utf-8",
                on_bad_lines="skip",
            )

            # Bỏ dòng đầu nếu là header chữ (khi file có header)
            if df.iloc[0]["video"] == "video":
                df = df.iloc[1:].reset_index(drop=True)

            self._df = df
            self._mtime = mtime
            logger.debug(f"OCR CSV loaded: {len(df)} rows from {self._csv_path}")
        except Exception as e:
            logger.error(f"Failed to load OCR CSV: {e}")
            self._df = None

        return self._df

    def _enrich_with_timing(self, res: Dict[str, Any]) -> None:
        """Thêm fps, frame_idx_video, time_str vào result dict (in-place)."""
        video_name = res.get("video_name", "")
        fps = load_fps_from_map(video_name)
        if not fps:
            return

        res["fps"] = fps

        frame_str = res.get("frame_idx", "")
        try:
            n = int(frame_str)
        except (TypeError, ValueError):
            n = parse_frame_number_from_filename(str(frame_str))

        if n is None:
            return

        frame_idx_video = load_frame_idx_from_map(video_name, n)
        if frame_idx_video is not None:
            res["frame_idx_video"] = frame_idx_video
            res["time_str"] = frame_to_timestr(frame_idx_video, fps)

    @staticmethod
    def _parse_json_col(value) -> Any:
        """Parse JSON string column, trả về raw value nếu lỗi."""
        if pd.isna(value):
            return []
        try:
            return json.loads(str(value))
        except Exception:
            return str(value)


# ── Module-level singleton (lazy init) ───────────────────────────────

_ocr_search_engine: Optional[OCRSearchEngine] = None


def get_ocr_search_engine() -> OCRSearchEngine:
    """Trả về singleton OCRSearchEngine, tạo nếu chưa có."""
    global _ocr_search_engine
    if _ocr_search_engine is None:
        _ocr_search_engine = OCRSearchEngine()
    return _ocr_search_engine
