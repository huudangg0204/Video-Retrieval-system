"""core/search.py — Text and image retrieval against the embedding index.

Fixes vs. original:
- frame_idx conversion to int wrapped in try/except (was silent ValueError).
- Object-filter threshold sign convention documented and preserved exactly.
- search_text_all threshold parameter now explicit.
- media_info lookup uses the shared util instead of repeating inline logic.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import config
from utils.frame_utils import parse_frame_number_from_filename
from utils.media_utils import (
    frame_to_timestr,
    load_fps_from_map,
    load_frame_idx_from_map,
    load_media_info,
)
from utils.file_utils import safe_basename
from .embeddings import EmbeddingIndex
from .models import ModelRegistry
from .vector_db import VectorDatabase


class SearchEngine:
    """
    Text/image/threshold search over a pre-built EmbeddingIndex.
    Stores results in the VectorDatabase for later replay.
    """

    def __init__(
        self,
        models: ModelRegistry,
        index: EmbeddingIndex,
        vector_db: VectorDatabase,
    ) -> None:
        self.models    = models
        self.index     = index
        self.vector_db = vector_db

    # ── Public search API ─────────────────────────────────────────────────────

    def search_text(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        save_to_db: bool = True,
        objects: str = "",
        threshold: float = config.DEFAULT_OBJ_THRESHOLD,
    ) -> List[Dict]:
        """
        Retrieve the top-k keyframes most similar to *query*.

        If *objects* is non-empty, additionally filter results by object detection:
          - threshold >= 0: keep frames where ANY desired object is detected.
          - threshold < 0 : keep frames where NONE of the desired objects appears.
        """
        if len(self.index.embeddings) == 0:
            print("❌ No embeddings loaded")
            return []

        text_emb = self.index.encode_text(query)
        sims     = self.index.cosine_similarity(text_emb, self.index.embeddings)[0]
        idx, scores = self.index.topk_indices(sims, top_k)
        results, sel_embs = self._build_results(idx, scores)

        if objects:
            results, sel_embs = self._apply_object_filter(results, sel_embs, objects, threshold)

        # Enrich with timing metadata
        for res in results:
            self._enrich_with_timing(res)

        if save_to_db and results:
            self.vector_db.add_query_results(
                query, results, np.asarray(sel_embs, dtype=np.float32)
            )

        return results

    def search_text_all(
        self,
        query: str,
        score: float = config.DEFAULT_SCORE_THRESHOLD,
        save_to_db: bool = True,
    ) -> List[Dict]:
        """Return ALL keyframes with similarity >= *score*, sorted desc."""
        if len(self.index.embeddings) == 0:
            print("❌ No embeddings loaded")
            return []

        text_emb = self.index.encode_text(query)
        sims     = self.index.cosine_similarity(text_emb, self.index.embeddings)[0]
        idx, scores = self.index.threshold_indices(sims, score)

        if len(scores) == 0:
            return []

        results, sel_embs = self._build_results(idx, scores)

        if save_to_db and results:
            self.vector_db.add_query_results(
                query, results, np.asarray(sel_embs, dtype=np.float32)
            )

        return results

    def search_image(
        self,
        image_path: str,
        top_k: int = config.DEFAULT_TOP_K,
        save_to_db: bool = True,
    ) -> List[Dict]:
        """Retrieve the top-k keyframes most visually similar to *image_path*."""
        if len(self.index.embeddings) == 0:
            print("❌ No embeddings loaded")
            return []

        img_emb = self.index.encode_image(image_path)
        sims    = self.index.cosine_similarity(img_emb, self.index.embeddings)[0]
        idx, scores = self.index.topk_indices(sims, top_k)
        query_tag   = f"[IMAGE] {safe_basename(image_path)}"
        results, sel_embs = self._build_results(idx, scores)

        if save_to_db and results:
            self.vector_db.add_query_results(query_tag, results, sel_embs)

        return results

    def get_frames_of_video(self, video_name: str) -> List[Dict]:
        """Return all keyframe dicts for a given video (no similarity score)."""
        folder = os.path.join(self.index.keyframes_dir, video_name)
        if not os.path.exists(folder):
            return []
        frames = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        return [
            {
                "image_path": os.path.join(folder, f),
                "video_name": video_name,
                "frame_idx":  os.path.splitext(f)[0],
                "similarity": None,
            }
            for f in frames
        ]

    def detect_objects(
        self,
        image_path: str,
        threshold: float = config.DEFAULT_OBJ_THRESHOLD,
    ) -> List[str]:
        """
        Run Faster R-CNN on *image_path* and return detected category names
        with confidence >= *threshold*.
        """
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ detect_objects: cannot open {image_path}: {e}")
            return []

        import torch
        tensor_img = self.models.detection_transform(img).to(self.models.device)
        with torch.no_grad():
            preds = self.models.detector([tensor_img])

        pred_scores = preds[0].get("scores", [])
        pred_labels = preds[0].get("labels", [])
        return [
            config.COCO_INSTANCE_CATEGORY_NAMES[label].lower()
            for score, label in zip(pred_scores, pred_labels)
            if score >= threshold
        ]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_results(
        self,
        indices: np.ndarray,
        scores: np.ndarray,
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Convert raw FAISS indices + scores into result dicts.
        Loads media_info once per video to avoid repeated disk reads.
        """
        results:   List[Dict]      = []
        sel_embs:  List[np.ndarray] = []
        _media_cache: Dict[str, dict] = {}

        for i, idx in enumerate(indices):
            img_path   = self.index.image_paths[idx]
            video_name = self.index.video_names[idx]
            frame_str  = os.path.splitext(os.path.basename(img_path))[0]

            if video_name not in _media_cache:
                _media_cache[video_name] = load_media_info(video_name)
            media_info = _media_cache[video_name]

            results.append({
                "image_path": img_path,
                "video_name": video_name,
                "similarity": float(scores[i]),
                "frame_idx":  frame_str,
                "watch_url":  media_info.get("watch_url", ""),
            })
            sel_embs.append(self.index.embeddings[idx])

        return results, np.asarray(sel_embs, dtype=np.float32) if sel_embs else np.array([])

    def _apply_object_filter(
        self,
        results:   List[Dict],
        sel_embs:  np.ndarray,
        objects:   str,
        threshold: float,
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Filter result/embedding pairs by detected objects.
        threshold >= 0 → keep frames that contain ANY desired object.
        threshold <  0 → keep frames that contain NONE of the desired objects.
        """
        desired = [o.strip().lower() for o in objects.split(",") if o.strip()]
        filtered_res: List[Dict]       = []
        filtered_emb: List[np.ndarray] = []

        for res, emb in zip(results, sel_embs):
            detected = self.detect_objects(res["image_path"], abs(threshold))
            if threshold >= 0:
                keep = any(obj in detected for obj in desired)
            else:
                keep = all(obj not in detected for obj in desired)
            if keep:
                filtered_res.append(res)
                filtered_emb.append(emb)

        emb_arr = np.asarray(filtered_emb, dtype=np.float32) if filtered_emb else np.array([])
        return filtered_res, emb_arr

    def _enrich_with_timing(self, res: Dict) -> None:
        """Add fps, frame_idx_video, and time_str to a result dict in-place."""
        video_name = res.get("video_name", "")
        fps = load_fps_from_map(video_name)
        if not fps:
            return
        res["fps"] = fps

        # frame_idx is the keyframe filename stem (e.g. "042")
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
            res["time_str"]        = frame_to_timestr(frame_idx_video, fps)


