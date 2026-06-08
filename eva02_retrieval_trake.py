"""eva02_retrieval_trake.py — Public façade for the EVA02 retrieval system.

This module is the single entry-point imported by sever.py and the GUI.
It delegates all logic to the core/* sub-packages and re-exports the
config constants that external callers depend on.

Refactored from the original 1097-line monolith.
Key fixes applied in core/:
  - Duplicate Faster R-CNN initialization removed (core/models.py)
  - Duplicate export_trake definition removed (core/export.py)
  - score/num variable collisions in TRAKE fixed (core/trake.py)
  - frame_idx int cast guarded with try/except (core/search.py)
  - media_info loaded via shared util, no more locals() hack
  - VectorDatabase state is always consistent on load failure
  - csv.DictReader now uses newline='' on Windows
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from typing import Any, Dict, List, Optional

# Re-export config constants so existing callers don't break
from config import (
    EMBEDDING_DIR,
    KEYFRAMES_DIR,
    DB_PATH,
    OUTPUT_FOLDER,
    MAP_KEYFRAME_PATH,
    MEDIA_INFO_PATH,
    MODEL_NAME,
    MODEL_PRETRAINED,
    DEFAULT_TOP_K,
    GAP_KEYFRAME,
    COCO_INSTANCE_CATEGORY_NAMES,
)

from core.vector_db  import VectorDatabase
from core.models     import ModelRegistry
from core.embeddings import EmbeddingIndex
from core.search     import SearchEngine
from core.rerank     import Reranker
from core.trake      import TrakeEngine
from core.export     import Exporter
from utils.media_utils import (
    load_fps_from_map,
    load_frame_idx_from_map,
    frame_to_timestr,
)
from utils.frame_utils import parse_frame_number_from_filename
from utils.file_utils  import safe_basename


class EVA02ImageRetrieval:
    """
    Public API surface for the EVA02 keyframe retrieval system.

    Instantiate once; all heavy models are loaded in __init__.
    All search/export methods preserve the exact signatures used by sever.py and the GUI.
    """

    def __init__(
        self,
        embedding_dir: str = EMBEDDING_DIR,
        keyframes_dir: str = KEYFRAMES_DIR,
        db_path:       str = DB_PATH,
    ) -> None:
        self.embedding_dir = embedding_dir
        self.keyframes_dir = keyframes_dir

        # Load models once
        self._models   = ModelRegistry()

        # Build embedding index
        self._index    = EmbeddingIndex(self._models, embedding_dir, keyframes_dir)

        # Persistent query store
        self._db       = VectorDatabase(db_path)

        # Sub-engines
        self._searcher = SearchEngine(self._models, self._index, self._db)
        self._reranker = Reranker(self._models)
        self._trake    = TrakeEngine(self._searcher, self._reranker)
        self._exporter = Exporter()

    # ── Properties (for backward compat with sever.py) ────────────────────────

    @property
    def keyframes_dir(self) -> str:
        return self._keyframes_dir

    @keyframes_dir.setter
    def keyframes_dir(self, v: str) -> None:
        self._keyframes_dir = v

    # ── Search ────────────────────────────────────────────────────────────────

    def search_text(
        self,
        query:      str,
        top_k:      int   = DEFAULT_TOP_K,
        save_to_db: bool  = True,
        objects:    str   = "",
        threshold:  float = 0.5,
    ) -> List[Dict]:
        return self._searcher.search_text(query, top_k, save_to_db, objects, threshold)

    def search_text_all(
        self,
        query:      str,
        score:      float = 0.2,
        save_to_db: bool  = True,
    ) -> List[Dict]:
        return self._searcher.search_text_all(query, score, save_to_db)

    def search_image(
        self,
        image_path: str,
        top_k:      int  = DEFAULT_TOP_K,
        save_to_db: bool = True,
    ) -> List[Dict]:
        return self._searcher.search_image(image_path, top_k, save_to_db)

    def get_frames_of_video(self, video_name: str) -> List[Dict]:
        return self._searcher.get_frames_of_video(video_name)

    def detect_objects(self, image_path: str, threshold: float = 0.5) -> List[str]:
        return self._searcher.detect_objects(image_path, threshold)

    # ── Encoding (used by sever.py for image reranking) ───────────────────────

    def encode_image(self, image_path: str):
        return self._index.encode_image(image_path)

    def get_embedding(self, image_path: str):
        return self._index.get_embedding(image_path)

    # ── TRAKE ─────────────────────────────────────────────────────────────────

    def trake_closest(
        self,
        events:     List[str],
        top_k:      int = 200,
        candidates: int = 200,
    ) -> Optional[List[Dict]]:
        return self._trake.trake_closest(events, top_k, candidates)

    def trake_highest(
        self,
        events:     List[str],
        top_k:      int = 200,
        candidates: int = 200,
        option:     int = 0,
    ) -> Optional[List[Dict]]:
        return self._trake.trake_highest(events, top_k, candidates, option)

    # ── Rerank ────────────────────────────────────────────────────────────────

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        return self._reranker.rerank(query, candidates)

    # ── Export ────────────────────────────────────────────────────────────────

    def export_results(
        self,
        results:     List[Dict],
        output_name: str,
        const_value: Optional[int] = None,
    ) -> str:
        return self._exporter.export_results(results, output_name, const_value)

    def export_trake(
        self,
        trake_payload: List[Dict],
        output_name:   str,
    ) -> str:
        return self._exporter.export_trake(trake_payload, output_name)

    def export_submission(
        self,
        video_name:  str,
        frame_id:    int,
        gap:         int,
        const_value: Optional[int] = None,
        name:        Optional[str] = None,
    ) -> str:
        return self._exporter.export_submission(video_name, frame_id, gap, const_value, name)

    def export_database(
        self,
        output_path:   str = "./exported_vectors",
        query_indices: Optional[List[int]] = None,
    ) -> None:
        self._db.export_to_numpy(output_path, query_indices)

    # ── Grouping ──────────────────────────────────────────────────────────────

    def group_results_by_video(
        self,
        results:       List[Dict],
        sort_by:       str           = "frame",
        top_per_video: Optional[int] = 50,
        with_time:     bool          = True,
    ) -> Dict[str, List[Dict]]:
        """Group results by video name and optionally enrich with timing info."""
        from collections import defaultdict
        grouped: Dict[str, List[Dict]] = defaultdict(list)

        for r in results or []:
            v    = r.get("video_name", "")
            item = dict(r)

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

            if "similarity" in item and item["similarity"] is not None:
                try:
                    item["similarity"] = float(item["similarity"])
                except Exception:
                    pass

            grouped[v].append(item)

        for v, lst in grouped.items():
            if sort_by == "similarity":
                lst.sort(key=lambda x: (x.get("similarity") is None, -(x.get("similarity") or 0.0)))
            else:
                lst.sort(key=lambda x: (x.get("frame_idx_video") is None, x.get("frame_idx_video") or 0))
            if isinstance(top_per_video, int) and top_per_video > 0:
                grouped[v] = lst[:top_per_video]

        return dict(grouped)

    # ── Conveniences ──────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        import numpy as np
        embs = self._index.embeddings
        return {
            "total_images":    len(self._index.image_paths),
            "total_videos":    len(set(self._index.video_names)),
            "embedding_shape": tuple(embs.shape) if isinstance(embs, np.ndarray) else None,
        }

    def get_stored_queries(self) -> List[str]:
        return self._db.get_all_queries()

    def reload_query_results(self, query_idx: int) -> List[Dict]:
        data = self._db.get_query_results(query_idx)
        if not data:
            return []
        return [
            {
                "image_path": data["image_paths"][i],
                "video_name": data["video_names"][i],
                "similarity": float(data["similarities"][i]),
                "frame_idx":  data["frame_indices"][i],
            }
            for i in range(len(data["image_paths"]))
        ]

    def show_database_stats(self) -> None:
        self._db.show_stats()

    # Timing helpers (kept for backward-compat with any external callers)
    def load_fps_from_map(self, video_name: str):
        return load_fps_from_map(video_name)

    def load_frame_idx_from_map(self, video_name: str, n: int):
        return load_frame_idx_from_map(video_name, n)

    def frame_to_timestr(self, frame_idx: int, fps: float) -> str:
        return frame_to_timestr(frame_idx, fps)


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:
    engine = EVA02ImageRetrieval()
    engine.show_database_stats()
    print(engine.get_stats())


if __name__ == "__main__":
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
