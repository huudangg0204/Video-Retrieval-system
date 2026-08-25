"""
Public API facade for the AIC-26 retrieval system.

This is the single entry point imported by the API server and GUI.
Delegates all logic to the core sub-modules.

Migrated from: eva02_retrieval_trake.py
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np

from loguru import logger

from aic26.config import settings
from aic26.config.constants import EVA02_EMBEDDING_DIM

from aic26.core.models import ModelRegistry
from aic26.core.embeddings import EmbeddingIndex
from aic26.core.database import VectorDatabase
from aic26.core.search import SearchEngine
from aic26.core.rerank import Reranker
from aic26.core.trake import TrakeEngine
from aic26.core.export import Exporter

from aic26.utils.media_utils import (
    load_fps_from_map,
    load_frame_idx_from_map,
    frame_to_timestr,
)
from aic26.utils.frame_utils import parse_frame_number_from_filename


# Configure UTF-8 output for Windows compatibility
sys.stdout.reconfigure(encoding='utf-8')


class EVA02ImageRetrieval:
    """
    Public API surface for the EVA-02 keyframe retrieval system.
    
    Instantiate once; all heavy models are loaded in __init__.
    All search/export methods preserve compatibility with the API and GUI.
    
    Usage:
        retriever = EVA02ImageRetrieval()
        results = retriever.search_text("a red car", top_k=10)
        solutions = retriever.trake_highest(["event1", "event2"])
    """
    
    def __init__(
        self,
        embedding_dir: Optional[str] = None,
        keyframes_dir: Optional[str] = None,
        db_path: Optional[str] = None,
        clip32_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the retrieval system.

        Args:
            embedding_dir: Directory containing .npy embedding files
            keyframes_dir: Directory containing keyframe images
            db_path: Path to the FAISS database pickle file
            clip32_dir: Optional directory of CLIP ViT-B-32 .npy files. When provided
                       and populated, dual-stream weighted search is enabled.
        """
        self.embedding_dir = embedding_dir or str(settings.embedding_dir)
        self.keyframes_dir = keyframes_dir or str(settings.keyframes_dir)
        self.clip32_dir = clip32_dir or str(settings.clip32_embedding_dir)

        logger.info("Initializing EVA-02 Retrieval System...")

        # Load models once
        logger.info("Loading AI models...")
        self._models = ModelRegistry()

        # Build embedding index
        logger.info("Loading embedding index...")
        self._index = EmbeddingIndex(
            self._models,
            self.embedding_dir,
            self.keyframes_dir,
            clip32_dir=self.clip32_dir,
            faiss_index_path=str(settings.faiss_index_path),
        )

        # Persistent query store
        logger.info("Loading vector database...")
        self._db = VectorDatabase(db_path or str(settings.db_path))

        # Sub-engines
        self._searcher = SearchEngine(
            self._models,
            self._index,
            self._db,
            weight_eva02=settings.eva02_search_weight,
            weight_clip32=settings.clip32_search_weight,
            weight_bm25=settings.bm25_search_weight,
            pool_multiplier=settings.dual_search_pool_multiplier,
        )
        self._reranker = Reranker(self._models)
        self._trake = TrakeEngine(self._searcher, self._reranker)
        self._exporter = Exporter()

        streams = ["eva02"]
        if self._index.is_dual:
            streams.append("clip32")
        logger.info(f"Retrieval system ready (streams: {','.join(streams)}).")

        # Segment search engine is lazy-loaded on first call to search_segments()
        # to avoid loading BGE-M3 weights at startup when not needed.
        self._segment_engine = None
    
    # ── Search ────────────────────────────────────────────────────────
    
    def search_text(
        self,
        query: str,
        top_k: int = settings.default_top_k,
        save_to_db: bool = True,
        objects: str = "",
        threshold: float = settings.default_obj_threshold,
    ) -> List[Dict[str, Any]]:
        """
        Search keyframes by text query.

        Args:
            query: Text description to search for
            top_k: Number of results to return
            save_to_db: Whether to save results to database
            objects: Comma-separated object names to filter by
            threshold: Object detection confidence threshold

        Returns:
            List of result dicts with image_path, video_name, similarity, etc.
        """
        logger.info(f"Text search: '{query[:100]}...' (top_k={top_k})")
        return self._searcher.search_text(
            query, top_k, save_to_db, objects, threshold,
        )

    def search_segments(
        self,
        query: str,
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """
        Stage-1: Segment-level search using BGE-M3.

        Searches the pre-built segment FAISS index and returns matching
        video segments plus unique candidate video IDs for Stage-2 use.

        Args:
            query: Free-text query in Vietnamese or English.
            top_k: Maximum number of segments to return.

        Returns:
            Dict with:
              - "segments": list of dicts (video_id, start_time, end_time,
                description, category, keywords, entities, score)
              - "candidate_video_ids": unique video IDs ordered by rank
        """
        if self._segment_engine is None:
            from aic26.core.segment_search import get_segment_search_engine
            self._segment_engine = get_segment_search_engine()
        return self._segment_engine.search(query, top_k=top_k)

    def search_text_filtered(
        self,
        query: str,
        video_ids: List[str],
        top_k: int = settings.default_top_k,
        save_to_db: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Stage-2: Filtered keyframe search using EVA-02, restricted to the
        supplied list of candidate video IDs from Stage-1.

        Slices the embedding matrix to only the keyframes of the requested
        videos and performs a fast dot-product search (O(M) with M << N).

        Args:
            query: Text description to search for (Vi or En).
            video_ids: Candidate video names from Stage-1 segment search.
            top_k: Number of keyframe results to return.
            save_to_db: Whether to persist results to VectorDatabase.

        Returns:
            List of result dicts with image_path, video_name, similarity, etc.
        """
        logger.info(
            f"Filtered text search: '{query[:100]}' across "
            f"{len(video_ids)} video(s) (top_k={top_k})"
        )
        return self._searcher.search_text_filtered(
            query, video_ids, top_k, save_to_db,
        )
    
    def search_text_all(
        self,
        query: str,
        score: float = settings.default_score_threshold,
        save_to_db: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search all keyframes above a similarity threshold.
        
        Args:
            query: Text description to search for
            score: Minimum similarity score (0.0 to 1.0)
            save_to_db: Whether to save results to database
            
        Returns:
            List of all matching result dicts
        """
        return self._searcher.search_text_all(query, score, save_to_db)
    
    def search_image(
        self,
        image_path: str,
        top_k: int = settings.default_top_k,
        save_to_db: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search keyframes by image similarity.
        
        Args:
            image_path: Path to the query image
            top_k: Number of results to return
            save_to_db: Whether to save results to database
            
        Returns:
            List of result dicts sorted by visual similarity
        """
        logger.info(f"Image search: {image_path} (top_k={top_k})")
        return self._searcher.search_image(image_path, top_k, save_to_db)
    
    def get_frames_of_video(self, video_name: str) -> List[Dict[str, Any]]:
        """
        Get all keyframes for a specific video.
        
        Args:
            video_name: Name of the video
            
        Returns:
            List of frame dicts (without similarity scores)
        """
        return self._searcher.get_frames_of_video(video_name)
    
    def detect_objects(
        self,
        image_path: str,
        threshold: float = settings.default_obj_threshold,
    ) -> List[str]:
        """
        Detect objects in an image.
        
        Args:
            image_path: Path to the image
            threshold: Detection confidence threshold
            
        Returns:
            List of detected object category names
        """
        return self._searcher.detect_objects(image_path, threshold)
    
    # ── Encoding ──────────────────────────────────────────────────────
    
    def encode_image(self, image_path: str) -> 'np.ndarray':
        """Encode an image to EVA-02 embedding vector."""
        return self._index.encode_image(image_path)
    
    def get_embedding(self, image_path: str) -> Optional['np.ndarray']:
        """Get pre-computed embedding for an image path."""
        return self._index.get_embedding(image_path)
    
    # ── TRAKE ─────────────────────────────────────────────────────────
    
    def trake_closest(
        self,
        events: List[str],
        top_k: int = settings.trake_default_top_k,
        candidates: int = settings.trake_default_candidates,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        TRAKE: Find event sequence with minimum frame gaps.
        
        Args:
            events: List of event descriptions in temporal order
            top_k: Results per event
            candidates: Number of candidate solutions
            
        Returns:
            List of solution dicts or None if no valid sequence found
        """
        logger.info(f"TRAKE closest: {len(events)} events")
        return self._trake.trake_closest(events, top_k, candidates)
    
    def trake_highest(
        self,
        events: List[str],
        top_k: int = settings.trake_default_top_k,
        candidates: int = settings.trake_default_candidates,
        option: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        TRAKE: Find event sequence with maximum total similarity.
        
        Args:
            events: List of event descriptions in temporal order
            top_k: Results per event
            candidates: Number of candidate solutions
            option: 0 for standard, 1 for threshold-based
            
        Returns:
            List of solution dicts or None if no valid sequence found
        """
        logger.info(f"TRAKE highest: {len(events)} events (option={option})")
        return self._trake.trake_highest(events, top_k, candidates, option)
    
    # ── Rerank ────────────────────────────────────────────────────────
    
    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank candidates using BLIP ITM."""
        return self._reranker.rerank(query, candidates)
    
    # ── Export ────────────────────────────────────────────────────────
    
    def export_results(
        self,
        results: List[Dict[str, Any]],
        output_name: str,
        const_value: Optional[Union[str, int]] = None,
    ) -> str:
        """Export search results to CSV submission file."""
        return self._exporter.export_results(results, output_name, const_value)
    
    def export_trake(
        self,
        trake_payload: List[Dict[str, Any]],
        output_name: str,
    ) -> str:
        """Export TRAKE results to CSV submission file."""
        return self._exporter.export_trake(trake_payload, output_name)
    
    def export_submission(
        self,
        video_name: str,
        frame_id: int,
        gap: int,
        const_value: Optional[Union[str, int]] = None,
        name: Optional[str] = None,
    ) -> str:
        """Export single-frame submission CSV."""
        return self._exporter.export_submission(video_name, frame_id, gap, const_value, name)
    
    def export_database(
        self,
        output_path: str = "./exported_vectors",
        query_indices: Optional[List[int]] = None,
    ) -> None:
        """Export vector database to numpy files."""
        self._db.export_to_numpy(output_path, query_indices)
    
    # ── Grouping ──────────────────────────────────────────────────────
    
    def group_results_by_video(
        self,
        results: List[Dict[str, Any]],
        sort_by: str = "frame",
        top_per_video: Optional[int] = 50,
        with_time: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group search results by video and enrich with timing info.
        
        Args:
            results: Search results to group
            sort_by: 'frame' or 'similarity'
            top_per_video: Limit results per video
            with_time: Add timing information
            
        Returns:
            Dict mapping video_name to list of enriched result dicts
        """
        from collections import defaultdict
        
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for r in results or []:
            v = r.get("video_name", "")
            item = dict(r)
            
            # Enrich with video frame index if available
            if item.get("frame_idx_video") is None:
                n = parse_frame_number_from_filename(str(item.get("frame_idx", "")))
                if n is not None:
                    idx = load_frame_idx_from_map(v, n)
                    if idx is not None:
                        item["frame_idx_video"] = idx
                        if with_time:
                            fps = load_fps_from_map(v)
                            if fps:
                                item["time_str"] = frame_to_timestr(idx, fps)
            
            # Ensure similarity is float
            if "similarity" in item and item["similarity"] is not None:
                try:
                    item["similarity"] = float(item["similarity"])
                except (TypeError, ValueError):
                    pass
            
            grouped[v].append(item)
        
        # Sort and limit
        for v, lst in grouped.items():
            if sort_by == "similarity":
                lst.sort(
                    key=lambda x: (
                        x.get("similarity") is None,
                        -(x.get("similarity") or 0.0),
                    )
                )
            else:
                lst.sort(
                    key=lambda x: (
                        x.get("frame_idx_video") is None,
                        x.get("frame_idx_video") or 0,
                    )
                )
            
            if isinstance(top_per_video, int) and top_per_video > 0:
                grouped[v] = lst[:top_per_video]
        
        return dict(grouped)
    
    # ── Convenience ───────────────────────────────────────────────────
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        import numpy as np
        embs = self._index.embeddings
        return {
            "total_images": len(self._index.image_paths),
            "total_videos": len(set(self._index.video_names)),
            "embedding_shape": tuple(embs.shape) if isinstance(embs, np.ndarray) else None,
            "embedding_dim": EVA02_EMBEDDING_DIM,
            "database_queries": len(self._db.get_all_queries()),
        }
    
    def get_stored_queries(self) -> List[str]:
        """Get list of stored query strings."""
        return self._db.get_all_queries()
    
    def reload_query_results(self, query_idx: int) -> List[Dict[str, Any]]:
        """Reload results from a stored query."""
        data = self._db.get_query_results(query_idx)
        if not data:
            return []
        return [
            {
                "image_path": data["image_paths"][i],
                "video_name": data["video_names"][i],
                "similarity": float(data["similarities"][i]),
                "frame_idx": data["frame_indices"][i],
            }
            for i in range(len(data["image_paths"]))
        ]
    
    def show_database_stats(self) -> None:
        """Display database statistics."""
        self._db.show_stats()
    
    # Timing helpers (backward compatibility)
    def load_fps_from_map(self, video_name: str) -> Optional[float]:
        return load_fps_from_map(video_name)
    
    def load_frame_idx_from_map(self, video_name: str, n: int) -> Optional[int]:
        return load_frame_idx_from_map(video_name, n)
    
    def frame_to_timestr(self, frame_idx: int, fps: float) -> str:
        return frame_to_timestr(frame_idx, fps)


# ── Module-level convenience ──────────────────────────────────────────

def main() -> None:
    """CLI entry point for testing."""
    engine = EVA02ImageRetrieval()
    engine.show_database_stats()
    print(engine.get_stats())

def run_gui() -> None:
    """Launch the desktop GUI."""
    from aic26.gui.retrieval_app import EVA02GUI
    app = EVA02GUI()
    app.run()


def run_filter_gui() -> None:
    """Launch the object filter GUI."""
    import tkinter as tk
    from aic26.gui.filter_app import ObjectFilterApp
    root = tk.Tk()
    app = ObjectFilterApp(root)
    root.mainloop()
    
    
if __name__ == "__main__":
    # Example TRAKE query
    query = [
        "A player is shooting a basketball",
        "A player is dunking a basketball",
        "A player is dribbling a basketball",
    ]
    engine = EVA02ImageRetrieval()
    engine.show_database_stats()
    print(engine.get_stats())
    
    results = engine.trake_highest(query, top_k=200, candidates=200, option=0)
    print(results)