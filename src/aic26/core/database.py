"""
Vector database for persistent query storage.

Stores query history with their results and embeddings.
Uses pickle for persistence and FAISS for similarity search.

Migrated from: core/vector_db.py
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from loguru import logger

from aic26.utils.file_utils import ensure_dir


class VectorDatabase:
    """
    Persistent storage for query history and results.
    
    Stores:
    - Query strings
    - Result embeddings (for FAISS search)
    - Result metadata (paths, scores, timestamps)
    
    Usage:
        db = VectorDatabase("faiss_db.pkl")
        db.add_query_results("query text", results, embeddings)
        db.save()
    """
    
    _EMPTY_DATA: Dict[str, Any] = {
        "queries": [],
        "embeddings": [],
        "image_paths": [],
        "video_names": [],
        "frame_indices": [],
        "similarities": [],
        "timestamps": [],
        "metadata": {},
    }
    
    def __init__(self, db_path: str) -> None:
        """
        Initialize the vector database.
        
        Args:
            db_path: Path to the pickle database file
        """
        self.db_path = db_path
        self._reset_data()
        self.index: Optional[faiss.IndexFlatIP] = None
        self._load()
    
    # ── Public API ────────────────────────────────────────────────
    
    def add_query_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add a new query and its results to the database.
        
        Args:
            query: The search query string
            results: List of result dicts
            embeddings: Float32 array of result embeddings
        """
        embs_f32 = embeddings.astype(np.float32)
        
        self.data["queries"].append(query)
        self.data["embeddings"].append(embs_f32)
        self.data["image_paths"].append([r["image_path"] for r in results])
        self.data["video_names"].append([r["video_name"] for r in results])
        self.data["frame_indices"].append([r["frame_idx"] for r in results])
        self.data["similarities"].append([r["similarity"] for r in results])
        self.data["timestamps"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Update FAISS index
        if self.index is None:
            d = embs_f32.shape[1]
            self.index = faiss.IndexFlatIP(d)
            logger.debug(f"Created FAISS IndexFlatIP with dimension {d}")
        self.index.add(embs_f32)
        
        logger.debug(f"Added query '{query[:50]}...' with {len(results)} results")
    
    def save(self) -> None:
        """Save the database to disk."""
        try:
            with open(self.db_path, "wb") as f:
                pickle.dump(self.data, f)
            logger.info(f"Database saved to {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
    
    def get_all_queries(self) -> List[str]:
        """Get list of all stored query strings."""
        return self.data["queries"]
    
    def get_query_results(self, query_idx: int) -> Optional[Dict[str, Any]]:
        """
        Get results for a specific query by index.
        
        Args:
            query_idx: Index of the query (0-based)
            
        Returns:
            Dict with query data, or None if index invalid
        """
        n = len(self.data["queries"])
        if not (0 <= query_idx < n):
            logger.warning(f"Query index {query_idx} out of range [0, {n})")
            return None
        
        return {
            "query": self.data["queries"][query_idx],
            "embeddings": self.data["embeddings"][query_idx],
            "image_paths": self.data["image_paths"][query_idx],
            "video_names": self.data["video_names"][query_idx],
            "frame_indices": self.data["frame_indices"][query_idx],
            "similarities": self.data["similarities"][query_idx],
            "timestamp": self.data["timestamps"][query_idx],
        }
    
    def get_combined_embeddings(
        self,
        query_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Combine embeddings from multiple queries.
        
        Args:
            query_indices: List of query indices (None = all)
            
        Returns:
            Tuple of (combined_embeddings, metadata_list)
        """
        if query_indices is None:
            query_indices = list(range(len(self.data["queries"])))
        
        combined_embs: List[np.ndarray] = []
        combined_meta: List[Dict[str, Any]] = []
        
        for idx in query_indices:
            if not (0 <= idx < len(self.data["queries"])):
                continue
            
            embs = self.data["embeddings"][idx]
            combined_embs.append(embs)
            
            for i in range(len(embs)):
                combined_meta.append({
                    "query_idx": idx,
                    "query": self.data["queries"][idx],
                    "image_path": self.data["image_paths"][idx][i],
                    "video_name": self.data["video_names"][idx][i],
                    "frame_idx": self.data["frame_indices"][idx][i],
                    "similarity": self.data["similarities"][idx][i],
                    "timestamp": self.data["timestamps"][idx],
                })
        
        if combined_embs:
            return np.concatenate(combined_embs, axis=0), combined_meta
        return np.array([]), []
    
    def export_to_numpy(
        self,
        output_path: str,
        query_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Export embeddings and metadata to files.
        
        Args:
            output_path: Base path for output files
            query_indices: List of query indices to export
        """
        embs, meta = self.get_combined_embeddings(query_indices)
        
        if len(embs) == 0:
            logger.warning("No embeddings to export")
            return
        
        ensure_dir(output_path)
        
        emb_path = f"{output_path}_embeddings.npy"
        meta_path = f"{output_path}_metadata.json"
        
        np.save(emb_path, embs)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(embs)} embeddings to {output_path}_*")
    
    def show_stats(self) -> None:
        """Display database statistics."""
        total_q = len(self.data["queries"])
        total_im = sum(len(p) for p in self.data["image_paths"])
        
        logger.info("=" * 50)
        logger.info("Vector Database Statistics")
        logger.info(f"  Total queries: {total_q}")
        logger.info(f"  Total images:  {total_im}")
        
        if total_q:
            logger.info("  Recent queries:")
            recent = list(zip(
                self.data["queries"][-5:],
                self.data["timestamps"][-5:],
            ))
            start = max(0, total_q - 5)
            for i, (q, ts) in enumerate(recent):
                logger.info(f"   [{start + i}] {q[:60]}... ({ts})")
        logger.info("=" * 50)
    
    # ── Private ───────────────────────────────────────────────────
    
    def _reset_data(self) -> None:
        """Initialize all data fields to empty state."""
        self.data = {
            k: (list(v) if isinstance(v, list) else dict(v))
            for k, v in self._EMPTY_DATA.items()
        }
    
    def _load(self) -> None:
        """Load database from disk and rebuild FAISS index."""
        try:
            with open(self.db_path, "rb") as f:
                loaded = pickle.load(f)
            
            # Normalize embeddings to list
            if isinstance(loaded.get("embeddings"), np.ndarray):
                loaded["embeddings"] = list(loaded["embeddings"])
            
            self.data = loaded
            
            # Rebuild FAISS index
            if self.data["embeddings"]:
                all_embs = np.vstack(self.data["embeddings"]).astype(np.float32)
                d = all_embs.shape[1]
                self.index = faiss.IndexFlatIP(d)
                self.index.add(all_embs)
            
            logger.info(f"Loaded database with {len(self.data['queries'])} queries")
            
        except FileNotFoundError:
            logger.info(f"No existing database at {self.db_path}. Starting fresh.")
        except Exception as e:
            logger.warning(f"Failed to load database ({e}). Starting fresh.")
            self._reset_data()