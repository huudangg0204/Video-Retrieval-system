"""
Search engine for AIC-26 video retrieval.

Provides text, image, and OCR search over Elasticsearch kNN index.
Replaces the legacy NumPy cosine similarity approach.

Architecture (from MemoriEase 3.0):
    Query → Encode (EVA-02 + BLIP-ITM) → ES kNN (weighted average) → Results

Usage:
    engine = SearchEngine(models, es_client)
    results = engine.search_text("a red car driving on highway", top_k=20)
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger

from aic26.config import settings
from aic26.core.elasticsearch_client import ElasticsearchClient
from aic26.core.embeddings import EmbeddingEncoder
from aic26.core.models import ModelRegistry
from aic26.utils.file_utils import safe_basename
from aic26.utils.media_utils import load_media_info


class SearchEngine:
    """
    Text and image search over Elasticsearch kNN index.

    Supports:
    - Dual-embedding text search (EVA-02 + BLIP-ITM weighted average)
    - Image-to-image search
    - Threshold-based search (all above cosine score)
    - Object detection filtering (via ES term filter)
    - OCR full-text search
    - Timing metadata enrichment

    Usage:
        engine = SearchEngine(models, es_client)
        results = engine.search_text("sunset over mountains", top_k=50)
    """

    def __init__(
        self,
        models: ModelRegistry,
        es_client: ElasticsearchClient,
        vector_db: Optional[Any] = None,   # kept for API compat, unused
    ) -> None:
        """
        Args:
            models: Loaded ModelRegistry.
            es_client: Connected ElasticsearchClient.
            vector_db: Deprecated — pass None. Kept for backward compat.
        """
        self.models = models
        self.es = es_client
        self._encoder = EmbeddingEncoder(models)

        if vector_db is not None:
            logger.warning(
                "SearchEngine: vector_db argument is deprecated in ES mode. "
                "Query history is now managed by ElasticsearchClient."
            )

        logger.debug("SearchEngine initialized (Elasticsearch backend).")

    # ── Public search API ─────────────────────────────────────────────────

    def search_text(
        self,
        query: str,
        top_k: int = settings.default_top_k,
        save_to_db: bool = True,
        objects: str = "",
        threshold: float = settings.default_obj_threshold,
    ) -> List[Dict[str, Any]]:
        """
        Search keyframes by text query using dual-embedding kNN.

        Args:
            query: Natural-language description.
            top_k: Number of results.
            save_to_db: Save query to ES query history.
            objects: Comma-separated object names to filter.
                     Prefix with '-' to exclude (e.g. '-person').
            threshold: Object filter confidence threshold (unused — ES filters
                       are based on pre-indexed detected_objects labels).

        Returns:
            List of result dicts with image_path, video_name, similarity, etc.
        """
        logger.debug(f"Text search: '{query[:80]}' (top_k={top_k})")

        object_filter, exclude_objects = _parse_object_filter(objects)

        if settings.use_dual_embedding:
            eva02_vec, blip_vec = self._encoder.encode_text_dual(query)
            hits = self.es.knn_search_dual(
                eva02_vec,
                blip_vec,
                top_k=top_k,
                object_filter=object_filter or None,
                exclude_objects=exclude_objects,
            )
        else:
            eva02_vec = self._encoder.encode_text_eva02(query)
            hits = self.es.knn_search_eva02(
                eva02_vec,
                top_k=top_k,
                object_filter=object_filter or None,
                exclude_objects=exclude_objects,
            )

        results = _hits_to_results(hits)

        if save_to_db and results:
            frame_ids = [f"{r['video_name']}__{r['frame_idx']}" for r in results]
            self.es.save_query(query, "text", frame_ids)

        logger.info(f"Text search '{query[:60]}': {len(results)} results")
        return results

    def search_text_all(
        self,
        query: str,
        score: float = settings.default_score_threshold,
        save_to_db: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Return all keyframes with cosine similarity above `score`.

        Uses single EVA-02 embedding with min_score filter.

        Args:
            query: Text description.
            score: Minimum similarity threshold (0.0–1.0).
            save_to_db: Save query to ES history.

        Returns:
            All matching result dicts.
        """
        eva02_vec = self._encoder.encode_text_eva02(query)

        # Fetch a large pool and filter by min_score
        hits = self.es.knn_search_eva02(
            eva02_vec,
            top_k=10000,
            min_score=score,
        )

        results = _hits_to_results(hits)

        if save_to_db and results:
            frame_ids = [f"{r['video_name']}__{r['frame_idx']}" for r in results]
            self.es.save_query(query, "text_all", frame_ids)

        logger.info(f"Text-all search (score>={score}): {len(results)} results")
        return results

    def search_image(
        self,
        image_path: str,
        top_k: int = settings.default_top_k,
        save_to_db: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search keyframes by image similarity (image-to-image).

        Uses dual-embedding if enabled.

        Args:
            image_path: Path to the query image.
            top_k: Number of results.
            save_to_db: Save query to ES history.

        Returns:
            List of result dicts sorted by similarity.
        """
        logger.debug(f"Image search: {image_path} (top_k={top_k})")

        if settings.use_dual_embedding:
            eva02_vec, blip_vec = self._encoder.encode_image_dual(image_path)
            hits = self.es.knn_search_dual(eva02_vec, blip_vec, top_k=top_k)
        else:
            eva02_vec = self._encoder.encode_image_eva02(image_path)
            hits = self.es.knn_search_eva02(eva02_vec, top_k=top_k)

        results = _hits_to_results(hits)

        if save_to_db and results:
            tag = f"[IMAGE] {safe_basename(image_path)}"
            frame_ids = [f"{r['video_name']}__{r['frame_idx']}" for r in results]
            self.es.save_query(tag, "image", frame_ids)

        logger.info(f"Image search: {len(results)} results")
        return results

    def search_ocr(
        self,
        text: str,
        top_k: int = settings.default_top_k,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search on OCR text extracted from keyframes.

        Args:
            text: OCR query string (e.g., "EXIT", "BUS STOP").
            top_k: Number of results.

        Returns:
            List of result dicts.
        """
        hits = self.es.text_search_ocr(text, top_k=top_k)
        results = _hits_to_results(hits, score_field="_score")
        logger.info(f"OCR search '{text}': {len(results)} results")
        return results

    def get_frames_of_video(self, video_name: str) -> List[Dict[str, Any]]:
        """
        Get all keyframe metadata for a video from ES.

        Args:
            video_name: Name of the video.

        Returns:
            List of frame dicts (no similarity scores).
        """
        resp = self.es._es.search(
            index=self.es.index_keyframes,
            body={
                "query": {"term": {"video_name": video_name}},
                "size": 10000,
                "sort": [{"frame_number": {"order": "asc"}}],
                "_source": {"excludes": ["eva02_embedding", "blip_embedding"]},
            },
        )
        results = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            results.append({
                "image_path": src.get("image_path", ""),
                "video_name": src.get("video_name", ""),
                "frame_idx": src.get("frame_idx", ""),
                "similarity": None,
            })
        return results

    def detect_objects(
        self,
        image_path: str,
        threshold: float = settings.default_obj_threshold,
    ) -> List[str]:
        """
        Detect objects in an image using Faster R-CNN.

        Args:
            image_path: Path to image.
            threshold: Confidence threshold.

        Returns:
            List of detected COCO category names.
        """
        from aic26.config.constants import COCO_CATEGORIES
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Cannot open image {image_path}: {e}")
            return []

        tensor_img = self.models.detection_transform(img).to(self.models.device)
        with torch.no_grad():
            preds = self.models.detector([tensor_img])

        pred_scores = preds[0].get("scores", [])
        pred_labels = preds[0].get("labels", [])

        return [
            COCO_CATEGORIES[label].lower()
            for score, label in zip(pred_scores, pred_labels)
            if score >= threshold
        ]


# ── Private helpers ───────────────────────────────────────────────────────────

def _hits_to_results(
    hits: List[Dict[str, Any]],
    score_field: str = "_score",
) -> List[Dict[str, Any]]:
    """
    Convert Elasticsearch hit dicts to the standard result format.

    Args:
        hits: List of ES hit dicts.
        score_field: Key to use as similarity score.

    Returns:
        List of result dicts with image_path, video_name, similarity, etc.
    """
    results = []
    _media_cache: Dict[str, dict] = {}

    for hit in hits:
        src = hit.get("_source", {})
        video_name = src.get("video_name", "")

        if video_name not in _media_cache:
            _media_cache[video_name] = load_media_info(video_name)
        media_info = _media_cache[video_name]

        results.append({
            "image_path": src.get("image_path", ""),
            "video_name": video_name,
            "frame_idx": src.get("frame_idx", ""),
            "similarity": float(hit.get(score_field, 0.0)),
            "fps": src.get("fps"),
            "frame_idx_video": src.get("frame_idx_video"),
            "time_str": src.get("time_str", ""),
            "watch_url": media_info.get("watch_url", src.get("watch_url", "")),
            "detected_objects": src.get("detected_objects", []),
            "ocr_text": src.get("ocr_text", ""),
        })

    return results


def _parse_object_filter(objects: str) -> Tuple[List[str], bool]:
    """
    Parse object filter string into (object_list, exclude_flag).

    Negative threshold in old API → now encoded as prefix '-' on any item.

    Args:
        objects: e.g. "person,bicycle" or "-person" to exclude.

    Returns:
        (list_of_objects, exclude_objects_flag)
    """
    if not objects:
        return [], False

    parts = [o.strip().lower() for o in objects.split(",") if o.strip()]
    if not parts:
        return [], False

    # If ALL parts start with '-', treat as exclusion filter
    if all(p.startswith("-") for p in parts):
        return [p.lstrip("-") for p in parts], True

    return [p.lstrip("-") for p in parts], False
