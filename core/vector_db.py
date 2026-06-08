"""core/vector_db.py — Persistent query-result store backed by pickle + FAISS.

Fixes vs. original:
- On load failure ALL fields are reset together (not just embeddings) to keep state consistent.
- Embeddings normalised to list on load to avoid downstream type errors.
- Cleaner separation between in-memory index rebuild and data loading.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from utils.file_utils import ensure_dir


class VectorDatabase:
    """Stores text/image queries together with their top-K result embeddings."""

    _EMPTY_DATA: Dict[str, Any] = {
        "queries":       [],
        "embeddings":    [],
        "image_paths":   [],
        "video_names":   [],
        "frame_indices": [],
        "similarities":  [],
        "timestamps":    [],
        "metadata":      {},
    }

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._reset_data()
        self.index: Optional[faiss.IndexFlatIP] = None
        self._load()

    # ── public API ────────────────────────────────────────────────────────────

    def add_query_results(
        self,
        query: str,
        results: List[Dict],
        embeddings: np.ndarray,
    ) -> None:
        """Append a new query + its results to the in-memory store and FAISS index."""
        embs_f32 = embeddings.astype(np.float32)
        self.data["queries"].append(query)
        self.data["embeddings"].append(embs_f32)
        self.data["image_paths"].append([r["image_path"]  for r in results])
        self.data["video_names"].append([r["video_name"]  for r in results])
        self.data["frame_indices"].append([r["frame_idx"] for r in results])
        self.data["similarities"].append([r["similarity"] for r in results])
        self.data["timestamps"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        if self.index is None:
            d = embs_f32.shape[1]
            self.index = faiss.IndexFlatIP(d)
        self.index.add(embs_f32)

    def save(self) -> None:
        try:
            with open(self.db_path, "wb") as f:
                pickle.dump(self.data, f)
            print(f"💾 Saved DB → {self.db_path}")
        except Exception as e:
            print(f"❌ DB save error: {e}")

    def get_all_queries(self) -> List[str]:
        return self.data["queries"]

    def get_query_results(self, query_idx: int) -> Optional[Dict[str, Any]]:
        n = len(self.data["queries"])
        if not (0 <= query_idx < n):
            return None
        return {
            "query":        self.data["queries"][query_idx],
            "embeddings":   self.data["embeddings"][query_idx],
            "image_paths":  self.data["image_paths"][query_idx],
            "video_names":  self.data["video_names"][query_idx],
            "frame_indices":self.data["frame_indices"][query_idx],
            "similarities": self.data["similarities"][query_idx],
            "timestamp":    self.data["timestamps"][query_idx],
        }

    def get_combined_embeddings(
        self,
        query_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, List[Dict]]:
        if query_indices is None:
            query_indices = list(range(len(self.data["queries"])))

        combined_embs: List[np.ndarray] = []
        combined_meta: List[Dict]       = []

        for idx in query_indices:
            if not (0 <= idx < len(self.data["queries"])):
                continue
            embs = self.data["embeddings"][idx]
            combined_embs.append(embs)
            for i in range(len(embs)):
                combined_meta.append({
                    "query_idx":  idx,
                    "query":      self.data["queries"][idx],
                    "image_path": self.data["image_paths"][idx][i],
                    "video_name": self.data["video_names"][idx][i],
                    "frame_idx":  self.data["frame_indices"][idx][i],
                    "similarity": self.data["similarities"][idx][i],
                    "timestamp":  self.data["timestamps"][idx],
                })

        if combined_embs:
            return np.concatenate(combined_embs, axis=0), combined_meta
        return np.array([]), []

    def export_to_numpy(
        self,
        output_path: str,
        query_indices: Optional[List[int]] = None,
    ) -> None:
        embs, meta = self.get_combined_embeddings(query_indices)
        if len(embs) == 0:
            print("❌ No embeddings to export")
            return
        ensure_dir(output_path)
        np.save(f"{output_path}_embeddings.npy", embs)
        with open(f"{output_path}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"📤 Exported: {len(embs)} embeddings → {output_path}_*")

    def show_stats(self) -> None:
        total_q  = len(self.data["queries"])
        total_im = sum(len(p) for p in self.data["image_paths"])
        print("\n📊 Vector DB Stats")
        print(f"  Total queries: {total_q}")
        print(f"  Total images:  {total_im}")
        if total_q:
            print("  Recent queries:")
            recent = list(zip(self.data["queries"][-5:], self.data["timestamps"][-5:]))
            start  = max(0, total_q - 5)
            for i, (q, ts) in enumerate(recent):
                print(f"   [{start + i}] {q}  ({ts})")

    # ── private ───────────────────────────────────────────────────────────────

    def _reset_data(self) -> None:
        """Initialise all data fields to empty — keeps them in sync."""
        self.data: Dict[str, Any] = {k: (list(v) if isinstance(v, list) else dict(v))
                                     for k, v in self._EMPTY_DATA.items()}

    def _load(self) -> None:
        try:
            with open(self.db_path, "rb") as f:
                loaded = pickle.load(f)

            # Normalise embeddings to list (old saves may store np.ndarray)
            if isinstance(loaded.get("embeddings"), np.ndarray):
                loaded["embeddings"] = list(loaded["embeddings"])

            self.data = loaded

            # Rebuild FAISS index from stored embeddings
            if self.data["embeddings"]:
                all_embs = np.vstack(self.data["embeddings"]).astype(np.float32)
                d = all_embs.shape[1]
                self.index = faiss.IndexFlatIP(d)
                self.index.add(all_embs)

            print(f"✅ Loaded DB with {len(self.data['queries'])} queries from {self.db_path}")

        except FileNotFoundError:
            print(f"ℹ️  No existing DB at {self.db_path}. Starting fresh.")
        except Exception as e:
            print(f"⚠️  Failed to load DB ({e}). Starting fresh.")
            self._reset_data()   # reset ALL fields, not just embeddings
