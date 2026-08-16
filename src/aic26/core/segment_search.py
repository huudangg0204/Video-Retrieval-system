"""
Segment Search Engine for AIC-26 (Stage-1 of hierarchical two-stage search).

Loads the pre-built segment_index.pkl and performs fast FAISS-based retrieval
of video segments given a free-text query (Vi/En supported via BGE-M3).

Returns top-K segments + unique candidate video IDs for Stage-2 filtering.

References:
    - Spec: docs/specs/two-stage-hierarchical-search.md §4.2
"""
from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from aic26.config import settings


class SegmentSearchEngine:
    """
    Stage-1 retrieval engine: text query → top-K segments → candidate video IDs.

    Loads segment_index.pkl (built by SegmentIndexBuilder) and performs
    FAISS IndexFlatIP search using BGE-M3 query embeddings.

    Usage:
        engine = SegmentSearchEngine(
            segment_index_path="data/faiss/segment_index.pkl"
        )
        result = engine.search("người sửa xe đạp", top_k=20)
        # result["segments"]           -> list of segment dicts with scores
        # result["candidate_video_ids"] -> unique video IDs ordered by rank
    """

    def __init__(
        self,
        segment_index_path: Optional[str] = None,
        model_name: str = settings.bge_m3_model_name,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the Segment Search Engine.

        Args:
            segment_index_path: Path to segment_index.pkl. Defaults to settings.
            model_name: BGE-M3 model id (passed to BGEM3TextEmbedder).
            device: Inference device. Auto-detected if None.
        """
        self._index_path = segment_index_path or str(settings.segment_index_path)
        self._model_name = model_name
        self._device = device

        # Index data (loaded lazily in _load_index)
        self._faiss_index = None
        self._embeddings: Optional[np.ndarray] = None
        self._segments: List[Dict[str, Any]] = []
        self._video_to_segment_indices: Dict[str, List[int]] = {}

        # Embedder (shared singleton via get_bge_m3_embedder)
        self._embedder = None

        self._load_index()

    # ── Public API ────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> Dict[str, Any]:
        """
        Search for video segments matching the given query.

        Args:
            query: Free-text query in Vietnamese or English.
            top_k: Maximum number of segments to return.

        Returns:
            Dict with:
              - "segments": list of dicts (video_id, start_time, end_time,
                description, category, keywords, entities, score)
              - "candidate_video_ids": unique video IDs ordered by first
                appearance in top-k results (preserves rank order)
        """
        if self._faiss_index is None:
            logger.error("Segment FAISS index not loaded. Run 'build-segment-index' first.")
            return {"segments": [], "candidate_video_ids": []}

        # Lazy-init embedder
        if self._embedder is None:
            from aic26.core.text_embedding import get_bge_m3_embedder
            self._embedder = get_bge_m3_embedder(self._model_name, self._device)

        # Embed query
        query_vec = self._embedder.embed_query(query)  # (1, 1024)

        # FAISS search
        k = min(top_k, self._faiss_index.ntotal)
        if k <= 0:
            return {"segments": [], "candidate_video_ids": []}

        q = query_vec.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        scores_raw, indices = self._faiss_index.search(q, k)
        scores_raw = scores_raw[0]
        indices = indices[0]

        # Build result list; track unique video IDs preserving rank order
        result_segments: List[Dict[str, Any]] = []
        seen_video_ids: dict = {}  # ordered-insertion dict (Python 3.7+ guaranteed)

        for score, idx in zip(scores_raw, indices):
            if idx < 0 or idx >= len(self._segments):
                continue
            seg = self._segments[idx]
            result_segments.append({
                "video_id": seg["video_id"],
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "description": seg["description"],
                "category": seg["category"],
                "keywords": seg["keywords"],
                "entities": seg["entities"],
                "score": float(score),
            })
            seen_video_ids[seg["video_id"]] = None  # key insertion preserves order

        return {
            "segments": result_segments,
            "candidate_video_ids": list(seen_video_ids.keys()),
        }

    @property
    def is_loaded(self) -> bool:
        """True if the segment index was successfully loaded."""
        return self._faiss_index is not None

    @property
    def total_segments(self) -> int:
        """Total number of segments in the index."""
        if self._faiss_index is None:
            return 0
        return self._faiss_index.ntotal

    @property
    def total_videos(self) -> int:
        """Total number of unique videos indexed."""
        return len(self._video_to_segment_indices)

    # ── Private ───────────────────────────────────────────────────────

    def _load_index(self) -> None:
        """Load the segment index from disk."""
        index_path = Path(self._index_path)

        if not index_path.exists():
            logger.warning(
                f"Segment index not found at '{index_path}'. "
                "Run 'aic26 pipeline build-segment-index' to create it."
            )
            return

        try:
            with open(index_path, "rb") as f:
                data = pickle.load(f)

            self._faiss_index = data.get("faiss_index")
            self._embeddings = data.get("embeddings")
            self._segments = data.get("segments", [])
            self._video_to_segment_indices = data.get("video_to_segment_indices", {})

            logger.info(
                f"Segment index loaded: {self._faiss_index.ntotal} segments, "
                f"{len(self._video_to_segment_indices)} videos "
                f"(model: {data.get('model_name', 'unknown')})"
            )
        except Exception as e:
            logger.error(f"Failed to load segment index from '{index_path}': {e}")
            self._faiss_index = None


# ── Module-level singleton ─────────────────────────────────────────────────

_engine_instance: Optional[SegmentSearchEngine] = None
_engine_lock = threading.Lock()


def get_segment_search_engine(
    segment_index_path: Optional[str] = None,
    model_name: str = settings.bge_m3_model_name,
    device: Optional[str] = None,
) -> SegmentSearchEngine:
    """
    Return the shared SegmentSearchEngine singleton (thread-safe lazy init).

    Args:
        segment_index_path: Path to segment_index.pkl (only used on first call).
        model_name: BGE-M3 model id (only used on first call).
        device: Inference device (only used on first call).

    Returns:
        Shared SegmentSearchEngine instance.
    """
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = SegmentSearchEngine(
                    segment_index_path=segment_index_path,
                    model_name=model_name,
                    device=device,
                )
    return _engine_instance
