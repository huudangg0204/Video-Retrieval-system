"""
Query history store for AIC-26.

Replaces the legacy FAISS+pickle VectorDatabase with an
Elasticsearch-backed query history store.

Stores:
- Query text and search type
- Result frame IDs (video__frame format)
- Timestamps

Usage:
    store = QueryHistoryStore(es_client)
    qid = store.add_query("a red car", "text", results)
    queries = store.list_queries()
    data = store.get_query_results(0)   # index-based for backward compat
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from aic26.core.elasticsearch_client import ElasticsearchClient


class QueryHistoryStore:
    """
    Stores and retrieves search query history via Elasticsearch.

    Replaces the legacy VectorDatabase (faiss_db.pkl).
    Provides index-based access (get_query_results(idx)) for
    backward compatibility with the existing API.

    Usage:
        store = QueryHistoryStore(es_client)
        store.add_query("sunset", "text", results)
        print(store.list_queries())
    """

    def __init__(self, es_client: ElasticsearchClient) -> None:
        self._es = es_client
        self._es.ensure_queries_index()
        # Local cache of query IDs in insertion order (for index-based access)
        self._query_ids: List[str] = []
        self._sync_ids()

    # ── Public API ────────────────────────────────────────────────────────

    def add_query(
        self,
        query_text: str,
        search_type: str,
        results: List[Dict[str, Any]],
    ) -> str:
        """
        Save a search query and its results.

        Args:
            query_text: Original query string.
            search_type: 'text' | 'image' | 'trake' | 'ocr'.
            results: List of result dicts (must have 'video_name', 'frame_idx').

        Returns:
            ES document ID of the saved query.
        """
        frame_ids = [
            f"{r.get('video_name', '')}_{r.get('frame_idx', '')}"
            for r in results
        ]
        qid = self._es.save_query(query_text, search_type, frame_ids)
        self._query_ids.append(qid)
        logger.debug(f"Saved query [{qid}]: '{query_text[:60]}' ({len(results)} results)")
        return qid

    # ── add_query_results: backward-compat alias ──────────────────────────

    def add_query_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,   # ignored in ES mode
    ) -> None:
        """
        Backward-compatible method matching the old VectorDatabase API.

        The `embeddings` parameter is accepted but not stored — Elasticsearch
        handles vector storage internally.
        """
        self.add_query(query, "text", results)

    def get_all_queries(self) -> List[str]:
        """Return list of all stored query strings."""
        docs = self._es.list_queries(limit=1000)
        return [d.get("query_text", "") for d in docs]

    def get_query_results(
        self,
        query_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Get stored query by integer index (most-recent first order from cache).

        Args:
            query_idx: 0-based index.

        Returns:
            Dict with 'query', 'image_paths', 'video_names', 'frame_indices',
            'similarities', 'timestamp'; or None if not found.
        """
        self._sync_ids()
        if not (0 <= query_idx < len(self._query_ids)):
            logger.warning(f"Query index {query_idx} out of range")
            return None

        qid = self._query_ids[query_idx]
        doc = self._es.get_query(qid)
        if doc is None:
            return None

        # Parse frame_ids into separate lists
        frame_ids = doc.get("result_frame_ids", [])
        image_paths: List[str] = []
        video_names: List[str] = []
        frame_indices: List[str] = []

        for fid in frame_ids:
            if "__" in fid:
                vid, frame = fid.split("__", 1)
            else:
                vid, frame = fid, ""
            video_names.append(vid)
            frame_indices.append(frame)
            image_paths.append("")  # path not stored in history

        return {
            "query": doc.get("query_text", ""),
            "image_paths": image_paths,
            "video_names": video_names,
            "frame_indices": frame_indices,
            "similarities": [0.0] * len(frame_ids),
            "timestamp": doc.get("timestamp", ""),
        }

    def list_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return most recent stored queries."""
        return self._es.list_queries(limit=limit)

    def show_stats(self) -> None:
        """Print query history statistics."""
        docs = self._es.list_queries(limit=1000)
        logger.info("=" * 50)
        logger.info("Query History (Elasticsearch)")
        logger.info(f"  Total queries: {len(docs)}")
        if docs:
            logger.info("  Recent queries:")
            for i, d in enumerate(docs[:5]):
                logger.info(
                    f"   [{i}] {d.get('query_text', '')[:60]} "
                    f"({d.get('result_count', 0)} results, {d.get('timestamp', '')})"
                )
        logger.info("=" * 50)

    def save(self) -> None:
        """No-op. Elasticsearch persists data automatically."""
        logger.debug("QueryHistoryStore.save() called (noop in ES mode)")

    # ── Private ───────────────────────────────────────────────────────────

    def _sync_ids(self) -> None:
        """Sync local ID cache from ES (sorted by timestamp ascending)."""
        try:
            docs = self._es.list_queries(limit=1000)
            # list_queries returns desc order; reverse for index-based access
            self._query_ids = [d["id"] for d in reversed(docs)]
        except Exception as e:
            logger.debug(f"_sync_ids failed: {e}")


# ── Backward-compatibility shim ───────────────────────────────────────────────

class VectorDatabase:
    """
    Backward-compatibility shim for code that imports VectorDatabase.

    Wraps QueryHistoryStore. The FAISS index and pickle persistence
    are replaced by Elasticsearch.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        logger.warning(
            "VectorDatabase is deprecated. "
            "Using QueryHistoryStore (Elasticsearch) instead."
        )
        from aic26.core.elasticsearch_client import ElasticsearchClient
        self._store = QueryHistoryStore(ElasticsearchClient())
        # Expose store interface
        self.data: Dict[str, Any] = {"queries": []}

    def add_query_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        self._store.add_query_results(query, results, embeddings)

    def get_all_queries(self) -> List[str]:
        return self._store.get_all_queries()

    def get_query_results(self, query_idx: int) -> Optional[Dict[str, Any]]:
        return self._store.get_query_results(query_idx)

    def show_stats(self) -> None:
        self._store.show_stats()

    def save(self) -> None:
        self._store.save()