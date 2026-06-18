"""
Elasticsearch client wrapper for AIC-26.

Manages the connection to Elasticsearch and provides:
- Index lifecycle (create, delete, exists)
- Bulk document indexing
- kNN vector search with metadata filters
- Full-text OCR search
- Query history storage

Architecture (from MemoriEase 3.0 paper):
    Offline: keyframes → embeddings + metadata → ES index
    Online:  query → encode → kNN search → ranked results
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from elasticsearch import Elasticsearch, NotFoundError, helpers
from elasticsearch.exceptions import ConnectionError as ESConnectionError
from loguru import logger

from aic26.config import settings


# ── Index Mappings ────────────────────────────────────────────────────────────

def _build_keyframes_mapping(eva02_dim: int, blip_dim: int) -> Dict[str, Any]:
    """Build the Elasticsearch index mapping for keyframes."""
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                # ── Identity ─────────────────────────────────────────
                "video_name": {"type": "keyword"},
                "frame_idx": {"type": "keyword"},
                "frame_number": {"type": "integer"},
                "image_path": {"type": "keyword", "index": False},

                # ── EVA-02 CLIP embedding (primary) ───────────────────
                "eva02_embedding": {
                    "type": "dense_vector",
                    "dims": eva02_dim,
                    "index": True,
                    "similarity": "cosine",
                },

                # ── BLIP-ITM embedding (dual, Option C Lite) ──────────
                "blip_embedding": {
                    "type": "dense_vector",
                    "dims": blip_dim,
                    "index": True,
                    "similarity": "cosine",
                },

                # ── Timing metadata ───────────────────────────────────
                "fps": {"type": "float"},
                "frame_idx_video": {"type": "integer"},
                "time_str": {"type": "keyword"},
                "watch_url": {"type": "keyword", "index": False},

                # ── Object detection (keyword filter) ─────────────────
                "detected_objects": {"type": "keyword"},

                # ── OCR full-text search ──────────────────────────────
                "ocr_text": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {
                        "keyword": {"type": "keyword", "ignore_above": 512}
                    },
                },

                # ── BLIP caption description ──────────────────────────
                "description": {
                    "type": "text",
                    "analyzer": "english",
                },

                # ── Indexing timestamp ────────────────────────────────
                "indexed_at": {"type": "date"},
            }
        },
    }


def _build_queries_mapping() -> Dict[str, Any]:
    """Build the Elasticsearch index mapping for query history."""
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "query_text": {"type": "text"},
                "search_type": {"type": "keyword"},   # text | image | trake | ocr
                "result_count": {"type": "integer"},
                "result_frame_ids": {"type": "keyword"},
                "timestamp": {"type": "date"},
            }
        },
    }


# ── Client ────────────────────────────────────────────────────────────────────

class ElasticsearchClient:
    """
    Thin wrapper around the official elasticsearch-py client.

    Provides index lifecycle management, bulk indexing, and
    kNN search for the AIC-26 keyframe retrieval system.

    Usage:
        client = ElasticsearchClient()
        client.ensure_keyframes_index()
        client.bulk_index_keyframes(documents)
        results = client.knn_search_eva02(query_vec, top_k=100)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        scheme: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        index_keyframes: Optional[str] = None,
        index_queries: Optional[str] = None,
    ) -> None:
        """
        Initialize the Elasticsearch client.

        Falls back to `settings` values for all parameters.
        """
        _host = host or settings.es_host
        _port = port or settings.es_port
        _scheme = scheme or settings.es_scheme
        _user = username or settings.es_username
        _pass = password or settings.es_password

        self.index_keyframes = index_keyframes or settings.es_index_keyframes
        self.index_queries = index_queries or settings.es_index_queries

        # Build connection URL
        url = f"{_scheme}://{_host}:{_port}"

        kwargs: Dict[str, Any] = {"hosts": [url]}
        if _user and _pass:
            kwargs["basic_auth"] = (_user, _pass)

        self._es = Elasticsearch(**kwargs)
        logger.info(f"ElasticsearchClient → {url}")

    # ── Connection ────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Return True if Elasticsearch is reachable."""
        try:
            return self._es.ping()
        except ESConnectionError:
            return False

    def get_info(self) -> Dict[str, Any]:
        """Get cluster info dict."""
        return dict(self._es.info())

    # ── Index Lifecycle ───────────────────────────────────────────────────

    def ensure_keyframes_index(self, force_recreate: bool = False) -> None:
        """
        Create the keyframes index if it does not exist.

        Args:
            force_recreate: If True, delete existing index first.
        """
        eva02_dim = settings.es_eva02_dim
        blip_dim = settings.es_blip_dim

        if force_recreate and self._es.indices.exists(index=self.index_keyframes):
            self._es.indices.delete(index=self.index_keyframes)
            logger.warning(f"Deleted existing index: {self.index_keyframes}")

        if not self._es.indices.exists(index=self.index_keyframes):
            mapping = _build_keyframes_mapping(eva02_dim, blip_dim)
            self._es.indices.create(index=self.index_keyframes, body=mapping)
            logger.info(
                f"Created index '{self.index_keyframes}' "
                f"(eva02={eva02_dim}d, blip={blip_dim}d)"
            )
        else:
            logger.debug(f"Index '{self.index_keyframes}' already exists.")

    def ensure_queries_index(self) -> None:
        """Create the query history index if it does not exist."""
        if not self._es.indices.exists(index=self.index_queries):
            self._es.indices.create(
                index=self.index_queries,
                body=_build_queries_mapping(),
            )
            logger.info(f"Created index '{self.index_queries}'")

    def delete_keyframes_index(self) -> None:
        """Delete the keyframes index (destructive!)."""
        if self._es.indices.exists(index=self.index_keyframes):
            self._es.indices.delete(index=self.index_keyframes)
            logger.warning(f"Deleted index: {self.index_keyframes}")

    def index_exists(self, index: Optional[str] = None) -> bool:
        """Check if an index exists."""
        return bool(self._es.indices.exists(index=index or self.index_keyframes))

    # ── Indexing ──────────────────────────────────────────────────────────

    def index_keyframe(self, doc: Dict[str, Any]) -> None:
        """
        Index a single keyframe document.

        Args:
            doc: Document dict. Must contain 'video_name' and 'frame_idx'.
                 Embeddings should be Python lists or numpy arrays.
        """
        doc_id = f"{doc['video_name']}__{doc['frame_idx']}"
        doc = _prepare_doc(doc)
        self._es.index(index=self.index_keyframes, id=doc_id, document=doc)

    def bulk_index_keyframes(
        self,
        docs: List[Dict[str, Any]],
        chunk_size: int = 500,
    ) -> Dict[str, int]:
        """
        Bulk index a list of keyframe documents.

        Args:
            docs: List of document dicts.
            chunk_size: Number of documents per bulk request.

        Returns:
            Dict with 'success' and 'failed' counts.
        """
        def _actions() -> Any:
            for doc in docs:
                doc_id = f"{doc['video_name']}__{doc['frame_idx']}"
                yield {
                    "_index": self.index_keyframes,
                    "_id": doc_id,
                    "_source": _prepare_doc(doc),
                }

        success, errors = helpers.bulk(
            self._es,
            _actions(),
            chunk_size=chunk_size,
            raise_on_error=False,
        )
        if errors:
            logger.warning(f"Bulk index errors: {len(errors)} failures")
            for err in errors[:5]:
                logger.debug(f"  Error: {err}")

        logger.info(f"Bulk indexed {success} documents ({len(errors)} errors)")
        return {"success": success, "failed": len(errors)}

    # ── kNN Search ────────────────────────────────────────────────────────

    def knn_search_eva02(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        object_filter: Optional[List[str]] = None,
        exclude_objects: bool = False,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        kNN search using EVA-02 embedding field.

        Args:
            query_vector: 1-D float32 numpy array (768-dim).
            top_k: Number of results to return.
            object_filter: List of object class names to filter by.
            exclude_objects: If True, exclude frames WITH these objects.
            min_score: Minimum cosine similarity score (0.0–1.0).

        Returns:
            List of result dicts with ES _score and _source fields.
        """
        return self._knn_search(
            field="eva02_embedding",
            query_vector=query_vector,
            top_k=top_k,
            object_filter=object_filter,
            exclude_objects=exclude_objects,
            min_score=min_score,
        )

    def knn_search_blip(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        object_filter: Optional[List[str]] = None,
        exclude_objects: bool = False,
    ) -> List[Dict[str, Any]]:
        """kNN search using BLIP-ITM embedding field."""
        return self._knn_search(
            field="blip_embedding",
            query_vector=query_vector,
            top_k=top_k,
            object_filter=object_filter,
            exclude_objects=exclude_objects,
        )

    def knn_search_dual(
        self,
        eva02_vector: np.ndarray,
        blip_vector: np.ndarray,
        top_k: int = 100,
        eva02_weight: Optional[float] = None,
        blip_weight: Optional[float] = None,
        object_filter: Optional[List[str]] = None,
        exclude_objects: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Dual kNN search combining EVA-02 and BLIP scores via weighted average.

        Implements the weighted average approach from MemoriEase 3.0:
            score = w1 * sim_eva02 + w2 * sim_blip

        Args:
            eva02_vector: EVA-02 query vector (768-dim).
            blip_vector: BLIP-ITM query vector (768-dim).
            top_k: Results to return.
            eva02_weight: Weight for EVA-02 score (default: settings.eva02_search_weight).
            blip_weight: Weight for BLIP score (default: settings.blip2_search_weight).
            object_filter: Object class name filter.
            exclude_objects: Exclude frames with these objects.

        Returns:
            List of merged result dicts, sorted by combined score.
        """
        w1 = eva02_weight if eva02_weight is not None else settings.eva02_search_weight
        w2 = blip_weight if blip_weight is not None else settings.blip2_search_weight

        # Fetch more candidates to allow merging
        fetch_k = min(top_k * 3, 1000)

        eva02_hits = self.knn_search_eva02(
            eva02_vector,
            top_k=fetch_k,
            object_filter=object_filter,
            exclude_objects=exclude_objects,
        )
        blip_hits = self.knn_search_blip(
            blip_vector,
            top_k=fetch_k,
            object_filter=object_filter,
            exclude_objects=exclude_objects,
        )

        # Merge by doc ID with weighted average
        scores: Dict[str, float] = {}
        docs: Dict[str, Dict[str, Any]] = {}

        for hit in eva02_hits:
            doc_id = hit["_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + w1 * hit["_score"]
            docs[doc_id] = hit

        for hit in blip_hits:
            doc_id = hit["_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + w2 * hit["_score"]
            if doc_id not in docs:
                docs[doc_id] = hit

        # Sort by combined score, return top_k
        sorted_ids = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]
        merged = []
        for doc_id in sorted_ids:
            hit = dict(docs[doc_id])
            hit["_score"] = scores[doc_id]
            merged.append(hit)

        logger.debug(
            f"Dual kNN: eva02={len(eva02_hits)} + blip={len(blip_hits)} "
            f"→ merged {len(merged)} results (w1={w1}, w2={w2})"
        )
        return merged

    # ── Full-Text Search ──────────────────────────────────────────────────

    def text_search_ocr(
        self,
        text: str,
        top_k: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search on OCR text field.

        Args:
            text: OCR text query string.
            top_k: Number of results.

        Returns:
            List of matching ES hit dicts.
        """
        query = {
            "query": {
                "match": {
                    "ocr_text": {
                        "query": text,
                        "fuzziness": "AUTO",
                    }
                }
            },
            "size": top_k,
            "_source": {"excludes": ["eva02_embedding", "blip_embedding"]},
        }
        resp = self._es.search(index=self.index_keyframes, body=query)
        hits = resp["hits"]["hits"]
        logger.debug(f"OCR search '{text[:60]}': {len(hits)} hits")
        return hits

    # ── Query History ─────────────────────────────────────────────────────

    def save_query(
        self,
        query_text: str,
        search_type: str,
        result_frame_ids: List[str],
    ) -> str:
        """
        Save a search query to the query history index.

        Args:
            query_text: The original query string.
            search_type: One of 'text', 'image', 'trake', 'ocr'.
            result_frame_ids: List of 'video__frame' identifiers.

        Returns:
            The Elasticsearch document ID.
        """
        self.ensure_queries_index()
        doc = {
            "query_text": query_text,
            "search_type": search_type,
            "result_count": len(result_frame_ids),
            "result_frame_ids": result_frame_ids,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        resp = self._es.index(index=self.index_queries, document=doc)
        return resp["_id"]

    def list_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return most recent stored queries."""
        self.ensure_queries_index()
        resp = self._es.search(
            index=self.index_queries,
            body={
                "query": {"match_all": {}},
                "sort": [{"timestamp": {"order": "desc"}}],
                "size": limit,
                "_source": True,
            },
        )
        return [
            {"id": h["_id"], **h["_source"]}
            for h in resp["hits"]["hits"]
        ]

    def get_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored query by document ID."""
        try:
            resp = self._es.get(index=self.index_queries, id=query_id)
            return {"id": resp["_id"], **resp["_source"]}
        except NotFoundError:
            return None

    # ── Stats ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return stats about the keyframes index."""
        try:
            count_resp = self._es.count(index=self.index_keyframes)
            total_docs = count_resp["count"]

            # Count distinct videos via aggregation
            agg_resp = self._es.search(
                index=self.index_keyframes,
                body={
                    "size": 0,
                    "aggs": {
                        "unique_videos": {
                            "cardinality": {"field": "video_name"}
                        }
                    },
                },
            )
            total_videos = agg_resp["aggregations"]["unique_videos"]["value"]
        except Exception as e:
            # Index may not exist yet (before init-es-index is run)
            if "index_not_found" in str(e).lower() or "404" in str(e):
                logger.debug(f"Index '{self.index_keyframes}' not yet created")
            else:
                logger.warning(f"Could not fetch ES stats: {e}")
            total_docs = 0
            total_videos = 0

        return {
            "total_keyframes": total_docs,
            "total_videos": total_videos,
            "index_keyframes": self.index_keyframes,
            "index_queries": self.index_queries,
        }

    # ── Private ───────────────────────────────────────────────────────────

    def _knn_search(
        self,
        field: str,
        query_vector: np.ndarray,
        top_k: int,
        object_filter: Optional[List[str]] = None,
        exclude_objects: bool = False,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Internal kNN search over a dense_vector field.

        Uses Elasticsearch 8.x kNN API with optional term filter.
        """
        vec = query_vector.flatten().tolist()
        num_candidates = max(top_k * 3, settings.es_knn_num_candidates)

        knn_clause: Dict[str, Any] = {
            "field": field,
            "query_vector": vec,
            "k": top_k,
            "num_candidates": num_candidates,
        }

        if object_filter:
            if exclude_objects:
                # Exclude frames that have ANY of the listed objects
                knn_clause["filter"] = {
                    "bool": {
                        "must_not": [
                            {"terms": {"detected_objects": object_filter}}
                        ]
                    }
                }
            else:
                # Include only frames that have AT LEAST ONE of the listed objects
                knn_clause["filter"] = {
                    "terms": {"detected_objects": object_filter}
                }

        body: Dict[str, Any] = {
            "knn": knn_clause,
            "size": top_k,
            "_source": {"excludes": ["eva02_embedding", "blip_embedding"]},
        }

        if min_score is not None:
            body["min_score"] = min_score

        resp = self._es.search(index=self.index_keyframes, body=body)
        hits = resp["hits"]["hits"]
        logger.debug(f"kNN [{field}]: {len(hits)} results (top_k={top_k})")
        return hits


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prepare_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a document for indexing:
    - Convert numpy arrays to Python lists.
    - Set indexed_at timestamp if missing.
    """
    prepared = {}
    for k, v in doc.items():
        if isinstance(v, np.ndarray):
            prepared[k] = v.flatten().tolist()
        else:
            prepared[k] = v

    if "indexed_at" not in prepared:
        prepared["indexed_at"] = datetime.now(tz=timezone.utc).isoformat()

    return prepared
