"""
Search engine for AIC-26 video retrieval.

Provides text, image, and threshold-based search over the embedding index.
Includes object detection filtering and timing enrichment.

Migrated from: core/search.py
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

import numpy as np

from PIL import Image

import torch

from aic26.config import settings
from aic26.config.constants import COCO_CATEGORIES

from aic26.utils.frame_utils import parse_frame_number_from_filename
from aic26.utils.media_utils import (
    frame_to_timestr,
    load_fps_from_map,
    load_frame_idx_from_map,
    load_media_info,
)
from aic26.utils.file_utils import safe_basename

from aic26.core.embeddings import EmbeddingIndex
from aic26.core.models import ModelRegistry
from aic26.core.database import VectorDatabase
from aic26.core.bm25 import BM25CaptionIndex
from aic26.core.translation import get_translation_engine


class SearchEngine:
    """
    Text and image search over pre-built embeddings.
    
    Supports:
    - Text-to-image search
    - Image-to-image search
    - Threshold-based search (all above score)
    - Object detection filtering
    - Timing metadata enrichment
    
    Usage:
        engine = SearchEngine(models, index, db)
        results = engine.search_text("a red car", top_k=10)
    """
    
    def __init__(
        self,
        models: ModelRegistry,
        index: EmbeddingIndex,
        vector_db: VectorDatabase,
        weight_eva02: float = 0.6,
        weight_clip32: float = 0.4,
        weight_bm25: float = 0.2,
        pool_multiplier: int = 5,
    ) -> None:
        """
        Initialize the search engine.

        Args:
            models: ModelRegistry with loaded models
            index: EmbeddingIndex with pre-computed embeddings
            vector_db: VectorDatabase for storing query history
            weight_eva02: Weight of EVA-02 cosine similarity in the dual blend
            weight_clip32: Weight of CLIP ViT-B-32 cosine similarity in the dual blend
            pool_multiplier: top_k * multiplier candidates are pulled from each
                             stream before weighted blending
        """
        self.models = models
        self.index = index
        self.vector_db = vector_db
        self.weight_eva02 = float(weight_eva02)
        self.weight_clip32 = float(weight_clip32)
        self.weight_bm25 = float(weight_bm25)
        self.pool_multiplier = max(1, int(pool_multiplier))

        self.bm25_index = None
        if self.weight_bm25 > 0:
            self.bm25_index = BM25CaptionIndex(image_paths=self.index.image_paths)

        # Vi -> En translation. EVA-02 / CLIP-32 / BM25 captions are all
        # English-trained, so an untranslated Vietnamese query silently
        # degrades to near-random similarity. See core/translation.py for
        # backend details (Gemini API with local envit5 fallback, cached).
        self.translator = get_translation_engine()

        streams = ["eva02"]
        if self.index.is_dual:
            streams.append("clip32")
        if self.bm25_index and self.bm25_index.bm25 is not None:
            streams.append("bm25")
        logger.debug(
            f"SearchEngine initialized. "
            f"Index size: {len(self.index.embeddings)} embeddings "
            f"(streams: {','.join(streams)})"
        )
    
    # ── Public search API ─────────────────────────────────────────
    
    def search_text(
        self,
        query: str,
        top_k: int = settings.default_top_k,
        save_to_db: bool = True,
        objects: str = "",
        threshold: float = settings.default_obj_threshold,
    ) -> List[Dict[str, Any]]:
        """
        Search keyframes by text description.
        
        Args:
            query: Text description to search for
            top_k: Number of top results to return
            save_to_db: Whether to save results to database
            objects: Comma-separated object names for filtering
            threshold: Object detection confidence threshold
                       >= 0: keep frames with ANY of the objects
                       < 0:  keep frames with NONE of the objects
            
        Returns:
            List of result dicts with keys:
            - image_path, video_name, similarity, frame_idx
            - watch_url (if media info available)
            - fps, frame_idx_video, time_str (if mapping available)
        """
        if len(self.index) == 0:
            logger.error("No embeddings or index loaded. Cannot search.")
            return []

        logger.debug(f"Searching: '{query[:80]}...' (top_k={top_k})")

        # Translate Vietnamese -> English. No-op (cached, near-instant) for
        # queries that are already English — see looks_vietnamese().
        query_en = self.translator.translate(query)
        if query_en != query:
            logger.info(f"Translated query: '{query}' -> '{query_en}'")

        # Encode query and compute similarities
        text_emb = self.index.encode_text(query_en)
        text_emb_clip32: Optional[np.ndarray] = None
        if self.index.is_dual:
            text_emb_clip32 = self.index.encode_text_clip32(query_en)

        # Multi-stream blend: EVA-02 always, CLIP-32/BM25 when usable
        # (BM25 also needs the English form since captions are generated
        # in English by pipeline/captioning.py)
        idx, scores = self._multi_stream_topk_indices(
            query_en, text_emb, text_emb_clip32, top_k
        )

        # Build results
        results, sel_embs = self._build_results(idx, scores)

        # Apply object filter if specified
        if objects:
            n_before = len(results)
            results, sel_embs = self._apply_object_filter(
                results, sel_embs, objects, threshold
            )
            logger.debug(
                f"Object filter '{objects}': {n_before} → {len(results)} results"
            )

        # Enrich with timing metadata
        for res in results:
            self._enrich_with_timing(res)

        # Save to database
        if save_to_db and results:
            self.vector_db.add_query_results(
                query,
                results,
                np.asarray(sel_embs, dtype=np.float32),
            )

        logger.info(f"Found {len(results)} results for query: '{query[:60]}...'")
        return results
    
    def search_text_all(
        self,
        query: str,
        score: float = settings.default_score_threshold,
        save_to_db: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Return all keyframes with similarity above threshold.

        Args:
            query: Text description to search for
            score: Minimum similarity score (0.0 to 1.0)
            save_to_db: Whether to save results to database

        Returns:
            List of all matching result dicts
        """
        if len(self.index) == 0:
            logger.error("No embeddings or index loaded")
            return []

        query_en = self.translator.translate(query)
        if query_en != query:
            logger.info(f"Translated query: '{query}' -> '{query_en}'")

        text_emb = self.index.encode_text(query_en)
        if self.index.use_faiss:
            # Query FAISS index for all items
            n = len(self.index)
            idx_all, sims_all = self.index.search_faiss(text_emb, n, stream="eva02")
            if self.index.is_dual:
                text_emb_clip32 = self.index.encode_text_clip32(query_en)
                idx32, sims32 = self.index.search_faiss(text_emb_clip32, n, stream="clip32")
                eva_dict = dict(zip(idx_all, sims_all))
                clip_dict = dict(zip(idx32, sims32))
                sims = np.array(
                    [
                        eva_dict.get(i, 0.0) * self.weight_eva02 + clip_dict.get(i, 0.0) * self.weight_clip32
                        for i in range(n)
                    ],
                    dtype=np.float32,
                )
                idx, scores = self.index.threshold_indices(sims, score)
            else:
                keep = sims_all >= score
                idx, scores = idx_all[keep], sims_all[keep]
        else:
            logger.info(
                "ℹ️ [SEARCH MODE: NUMPY FALLBACK] Performing threshold search on in-memory .npy matrix."
            )
            sims = self.index.cosine_similarity(text_emb, self.index.embeddings)[0]
            if self.index.is_dual:
                text_emb_clip32 = self.index.encode_text_clip32(query_en)
                sims_clip32 = self.index.cosine_similarity(
                    text_emb_clip32, self.index.clip32_embeddings
                )[0]
                sims = sims * self.weight_eva02 + sims_clip32 * self.weight_clip32

            idx, scores = self.index.threshold_indices(sims, score)

        if len(scores) == 0:
            logger.info(f"No results above threshold {score}")
            return []

        results, sel_embs = self._build_results(idx, scores)

        if save_to_db and results:
            self.vector_db.add_query_results(
                query,
                results,
                np.asarray(sel_embs, dtype=np.float32),
            )

        logger.info(f"Found {len(results)} results above score {score}")
        return results
    
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
        if len(self.index) == 0:
            logger.error("No embeddings or index loaded")
            return []

        img_emb = self.index.encode_image(image_path)
        if self.index.is_dual:
            img_emb_clip32 = self.index.encode_image_clip32(image_path)
            idx, scores = self._dual_topk_indices(img_emb, img_emb_clip32, top_k)
        elif self.index.use_faiss:
            idx, scores = self.index.search_faiss(img_emb, top_k, stream="eva02")
        else:
            logger.info("ℹ️ [SEARCH MODE: NUMPY FALLBACK] Performing image similarity on in-memory .npy matrix.")
            sims = self.index.cosine_similarity(img_emb, self.index.embeddings)[0]
            idx, scores = self.index.topk_indices(sims, top_k)

        query_tag = f"[IMAGE] {safe_basename(image_path)}"
        results, sel_embs = self._build_results(idx, scores)

        if save_to_db and results:
            self.vector_db.add_query_results(query_tag, results, sel_embs)

        logger.info(f"Image search found {len(results)} results")
        return results
    
    def get_frames_of_video(self, video_name: str) -> List[Dict[str, Any]]:
        """
        Get all keyframe information for a video.
        
        Args:
            video_name: Name of the video
            
        Returns:
            List of frame dicts with image_path, video_name, frame_idx
        """
        folder = os.path.join(self.index.keyframes_dir, video_name)
        if not os.path.exists(folder):
            logger.warning(f"Keyframe folder not found: {folder}")
            return []
        
        frames = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        
        return [
            {
                "image_path": os.path.join(folder, f),
                "video_name": video_name,
                "frame_idx": os.path.splitext(f)[0],
                "similarity": None,
            }
            for f in frames
        ]

    def search_text_filtered(
        self,
        query: str,
        video_ids: List[str],
        top_k: int = settings.default_top_k,
        save_to_db: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Stage-2 filtered keyframe search: EVA-02 search restricted to a
        subset of video IDs supplied from Stage-1 segment retrieval.

        Instead of scanning the full corpus (potentially 500k+ keyframes),
        this method slices the embedding matrix to only the rows belonging
        to the requested videos and performs a fast numpy dot product.

        Args:
            query: Text query (Vi or En — translated automatically).
            video_ids: List of candidate video names from Stage-1.
            top_k: Number of keyframe results to return.
            save_to_db: Whether to persist results to VectorDatabase.

        Returns:
            List of result dicts (same schema as search_text):
            image_path, video_name, similarity, frame_idx, watch_url,
            fps, frame_idx_video, time_str (when available).
        """
        if len(self.index) == 0:
            logger.error("No embeddings or index loaded. Cannot search.")
            return []

        if not video_ids:
            logger.warning("search_text_filtered called with empty video_ids.")
            return []

        if not self.index.video_to_indices:
            logger.warning(
                "video_to_indices map is empty — index may not have been built correctly. "
                "Falling back to full search_text()."
            )
            return self.search_text(query, top_k=top_k, save_to_db=save_to_db)

        # Translate Vi -> En
        query_en = self.translator.translate(query)
        if query_en != query:
            logger.info(f"[Filtered] Translated: '{query}' -> '{query_en}'")

        # Encode query with EVA-02
        query_vec = self.index.encode_text(query_en)  # shape (1, 768)

        # Gather row indices for requested videos
        target_indices: List[int] = []
        missing = []
        for vid in video_ids:
            idxs = self.index.video_to_indices.get(vid)
            if idxs:
                target_indices.extend(idxs)
            else:
                missing.append(vid)

        if missing:
            logger.debug(
                f"[Filtered] {len(missing)} video(s) not found in index: {missing[:5]}"
            )

        if not target_indices:
            logger.warning(
                f"[Filtered] None of the supplied video_ids found in index: {video_ids[:5]}"
            )
            return []

        target_np = np.array(target_indices, dtype=np.int64)

        if len(self.index.embeddings) == 0:
            logger.error("Embedding matrix is empty; cannot perform filtered search.")
            return []

        # Slice embedding sub-matrix and compute dot products
        sub_embeddings = self.index.embeddings[target_np]  # (M, 768)
        scores = np.dot(sub_embeddings, query_vec.T).squeeze()  # (M,)
        if scores.ndim == 0:
            scores = scores.reshape(1)

        # Fast top-k via argpartition (O(M) instead of O(M log M))
        k = min(top_k, len(scores))
        if k <= 0:
            return []

        if len(scores) > k:
            top_sub_idx = np.argpartition(scores, -k)[-k:]
            top_sub_idx = top_sub_idx[np.argsort(-scores[top_sub_idx])]
        else:
            top_sub_idx = np.argsort(-scores)

        final_indices = target_np[top_sub_idx]         # original corpus indices
        final_scores = scores[top_sub_idx]

        logger.info(
            f"[Filtered] Searched {len(target_indices)} keyframes across "
            f"{len(video_ids)} video(s); returning top-{k} results."
        )

        # Build result dicts
        results, sel_embs = self._build_results(final_indices, final_scores)

        # Enrich with timing metadata
        for res in results:
            self._enrich_with_timing(res)

        if save_to_db and results:
            self.vector_db.add_query_results(
                f"[FILTERED] {query}",
                results,
                np.asarray(sel_embs, dtype=np.float32),
            )

        return results
    
    def detect_objects(
        self,
        image_path: str,
        threshold: float = settings.default_obj_threshold,
    ) -> List[str]:
        """
        Detect objects in an image using Faster R-CNN.
        
        Args:
            image_path: Path to the image
            threshold: Detection confidence threshold
            
        Returns:
            List of detected COCO category names (lowercase)
        """
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
        
        detected = [
            COCO_CATEGORIES[label].lower()
            for score, label in zip(pred_scores, pred_labels)
            if score >= threshold
        ]
        
        return detected
    
    # ── Private helpers ───────────────────────────────────────────


    def _build_results(
        self,
        indices: np.ndarray,
        scores: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Convert FAISS indices + scores to result dicts.

        Args:
            indices: Array of indices into the embedding index
            scores: Array of similarity scores

        Returns:
            Tuple of (results_list, selected_embeddings_array)
        """
        results: List[Dict[str, Any]] = []
        sel_embs: List[np.ndarray] = []
        _media_cache: Dict[str, dict] = {}

        for i, idx in enumerate(indices):
            img_path = self.index.image_paths[idx].replace(os.sep, "/")
            video_name = self.index.video_names[idx]

            # Cache media info per video
            if video_name not in _media_cache:
                _media_cache[video_name] = load_media_info(video_name)
            media_info = _media_cache[video_name]

            frame_str = os.path.splitext(os.path.basename(img_path))[0]

            results.append({
                "image_path": img_path,
                "video_name": video_name,
                "similarity": float(scores[i]),
                "frame_idx": frame_str,
                "watch_url": media_info.get("watch_url", ""),
            })
            if len(self.index.embeddings) > idx:
                sel_embs.append(self.index.embeddings[idx])
            else:
                sel_embs.append(np.zeros((768,), dtype=np.float32))

        emb_arr = np.asarray(sel_embs, dtype=np.float32) if sel_embs else np.array([])
        return results, emb_arr

    def _dual_topk_indices(
        self,
        query_emb_eva: np.ndarray,
        query_emb_clip32: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward-compatible wrapper for the dual-stream blend. Internally
        delegates to `_multi_stream_topk_indices`.
        """
        return self._multi_stream_topk_indices("", query_emb_eva, query_emb_clip32, top_k)

    def _multi_stream_topk_indices(
        self,
        query: str,
        query_emb_eva: np.ndarray,
        query_emb_clip32: np.ndarray | None,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weighted-average top-k over any combination of the available embedding
        streams (EVA-02, CLIP ViT-B-32, BM25).

        When FAISS index is present (self.index.use_faiss == True), candidate search
        is performed directly in C++ via FAISS IndexFlatIP.search().
        When FAISS index is missing (self.index.use_faiss == False), falls back
        to NumPy in-memory dot product and logs an informational warning.
        """
        n = len(self.index)
        if n == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        # ── Priority 1: Direct FAISS Index Search ─────────────────
        if self.index.use_faiss:
            logger.debug(f"[FAISS SEARCH] Executing query directly on FAISS IndexFlatIP (n={n})")
            all_candidates: List[np.ndarray] = []
            per_stream_scores: Dict[str, Dict[int, float]] = {}

            # Stream 1: EVA-02 via FAISS
            pool_k_eva = min(n, max(top_k * self.pool_multiplier, top_k))
            eva_idx, eva_scores = self.index.search_faiss(query_emb_eva, pool_k_eva, stream="eva02")
            all_candidates.append(eva_idx)
            per_stream_scores["eva02"] = dict(zip(eva_idx, [float(s) for s in eva_scores]))

            # Stream 2: CLIP-32 via FAISS (if dual active)
            if self.index.is_dual and query_emb_clip32 is not None:
                pool_k_clip = min(n, max(top_k * self.pool_multiplier, top_k))
                clip_idx, clip_scores = self.index.search_faiss(query_emb_clip32, pool_k_clip, stream="clip32")
                all_candidates.append(clip_idx)
                per_stream_scores["clip32"] = dict(zip(clip_idx, [float(s) for s in clip_scores]))

            # Stream 3: BM25 (if enabled)
            if query and self.bm25_index and self.bm25_index.bm25 is not None and self.weight_bm25 > 0:
                bm25_all = self.bm25_index.get_scores(query)
                pool_k_bm25 = min(n, max(top_k * self.pool_multiplier, top_k))
                bm25_top_idx, _ = self.index.topk_indices(bm25_all, pool_k_bm25)
                all_candidates.append(bm25_top_idx)
                per_stream_scores["bm25"] = {int(i): float(bm25_all[i]) for i in bm25_top_idx}

            cand_idx = np.unique(np.concatenate(all_candidates))

            # If embeddings matrix is present, compute exact cosine similarity for candidate union
            if len(self.index.embeddings) > 0:
                eva_cand_sims = np.dot(query_emb_eva, self.index.embeddings[cand_idx].T)[0]
                blend = self.weight_eva02 * eva_cand_sims.astype(np.float32)
                if self.index.is_dual and query_emb_clip32 is not None and len(self.index.clip32_embeddings) > 0:
                    clip_cand_sims = np.dot(query_emb_clip32, self.index.clip32_embeddings[cand_idx].T)[0]
                    blend += self.weight_clip32 * clip_cand_sims.astype(np.float32)
                if "bm25" in per_stream_scores:
                    bm25_cand_sims = bm25_all[cand_idx].astype(np.float32)
                    blend += self.weight_bm25 * bm25_cand_sims
            else:
                blend = np.zeros(len(cand_idx), dtype=np.float32)
                for i, idx in enumerate(cand_idx):
                    idx_int = int(idx)
                    s = self.weight_eva02 * per_stream_scores["eva02"].get(idx_int, 0.0)
                    if "clip32" in per_stream_scores:
                        s += self.weight_clip32 * per_stream_scores["clip32"].get(idx_int, 0.0)
                    if "bm25" in per_stream_scores:
                        s += self.weight_bm25 * per_stream_scores["bm25"].get(idx_int, 0.0)
                    blend[i] = s

            k = min(top_k, len(blend))
            if k <= 0:
                return np.array([], dtype=int), np.array([], dtype=np.float32)
            part = np.argpartition(blend, -k)[-k:]
            order = np.argsort(blend[part])[::-1]
            return cand_idx[part[order]], blend[part[order]]

        # ── Priority 2 (Fallback): In-Memory .npy Matrix Search ───
        logger.info(
            "ℹ️ [SEARCH MODE: NUMPY FALLBACK] Performing cosine similarity on in-memory .npy matrix. "
            "(FAISS index file is missing). Run 'aic26 pipeline build-index' to enable high-speed FAISS search."
        )

        streams = [
            (
                "eva02",
                self.weight_eva02,
                self.pool_multiplier,
                lambda: self.index.cosine_similarity(query_emb_eva, self.index.embeddings)[0],
            )
        ]
        if self.index.is_dual and query_emb_clip32 is not None:
            streams.append(
                (
                    "clip32",
                    self.weight_clip32,
                    self.pool_multiplier,
                    lambda: self.index.cosine_similarity(query_emb_clip32, self.index.clip32_embeddings)[0],
                )
            )
        if query and self.bm25_index and self.bm25_index.bm25 is not None and self.weight_bm25 > 0:
            streams.append(
                (
                    "bm25",
                    self.weight_bm25,
                    self.pool_multiplier,
                    lambda: self.bm25_index.get_scores(query),
                )
            )

        # Per-stream: compute sim, then top pool
        all_candidates = []
        per_stream_sims = {}
        for label, _w, mult, score_fn in streams:
            sims = score_fn()
            per_stream_sims[label] = sims
            pool_k = min(n, max(top_k * mult, top_k))
            top_idx, _ = self.index.topk_indices(sims, pool_k)
            all_candidates.append(top_idx)

        # Union of candidate indices across all streams
        cand_idx = np.unique(np.concatenate(all_candidates))

        # Blend scores on the candidate set, weighted per stream
        blend = np.zeros(len(cand_idx), dtype=np.float32)
        for label, weight, _mult, _score_fn in streams:
            blend += weight * per_stream_sims[label][cand_idx].astype(np.float32)

        # Top-k of the blended set
        k = min(top_k, len(blend))
        if k <= 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        part = np.argpartition(blend, -k)[-k:]
        order = np.argsort(blend[part])[::-1]
        return cand_idx[part[order]], blend[part[order]]
    
    def _apply_object_filter(
        self,
        results: List[Dict[str, Any]],
        sel_embs: np.ndarray,
        objects: str,
        threshold: float,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Filter results by detected objects using precomputed detections.
        
        Uses YOLOv26 detection data from metadata sidecars instead of
        running Faster R-CNN in real-time (saves ~170MB VRAM).
        
        Args:
            results: Search results to filter
            sel_embs: Corresponding embeddings
            objects: Comma-separated object names
            threshold: >= 0: keep if ANY object found
                      < 0:  keep if NONE found
            
        Returns:
            Filtered (results, embeddings) tuple
        """
        desired = [o.strip().lower() for o in objects.split(",") if o.strip()]
        filtered_res: List[Dict[str, Any]] = []
        filtered_emb: List[np.ndarray] = []
        
        for res, emb in zip(results, sel_embs):
            detected = self._get_precomputed_objects(
                res.get("video_name", ""),
                res.get("frame_idx", ""),
                abs(threshold),
            )
            
            if threshold >= 0:
                keep = any(obj in detected for obj in desired)
            else:
                keep = all(obj not in detected for obj in desired)
            
            if keep:
                filtered_res.append(res)
                filtered_emb.append(emb)
        
        emb_arr = np.asarray(filtered_emb, dtype=np.float32) if filtered_emb else np.array([])
        return filtered_res, emb_arr

    def _get_precomputed_objects(
        self, video_name: str, frame_idx: str, threshold: float
    ) -> List[str]:
        """
        Get detected object labels from precomputed metadata sidecars.

        Falls back to Faster R-CNN if no metadata exists (lazy-loaded).
        """
        import json as _json
        meta_path = os.path.join(
            str(settings.metadata_dir), video_name, f"{frame_idx}.json"
        )
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = _json.load(f)
            return [
                d["label"].lower()
                for d in meta.get("detections", [])
                if d.get("score", 0) >= threshold
            ]
        except (FileNotFoundError, KeyError):
            # Fallback: run Faster R-CNN (lazy-loaded, only if no metadata)
            img_path = os.path.join(
                str(settings.keyframes_dir), video_name, f"{frame_idx}.jpg"
            )
            return self.detect_objects(img_path, threshold)
    
    def _enrich_with_timing(self, res: Dict[str, Any]) -> None:
        """
        Add FPS, frame index, and time string to a result dict.
        
        Modifies the dict in-place.
        
        Args:
            res: Result dict to enrich
        """
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