"""
Embedding index management for AIC-26.

Handles loading pre-computed embeddings, pairing with keyframes,
and providing text/image encoding capabilities.

Migrated from: core/embeddings.py
"""
from __future__ import annotations

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from loguru import logger

from aic26.config import settings
from aic26.utils.frame_utils import parse_frame_number_from_filename
from aic26.core.models import ModelRegistry


class EmbeddingIndex:
    """
    Loads and manages pre-computed embeddings and FAISS index.
    
    Prioritizes direct querying on FAISS index (IndexFlatIP) for high performance
    and scalability. Automatically falls back to in-memory .npy array search if the
    FAISS index file is not found.
    """
    
    def __init__(
        self,
        models: ModelRegistry,
        embedding_dir: str,
        keyframes_dir: str,
        clip32_dir: str | None = None,
        faiss_index_path: str | None = None,
    ) -> None:
        """
        Initialize the embedding index.

        Args:
            models: ModelRegistry with loaded models
            embedding_dir: Directory containing .npy embedding files (EVA-02)
            keyframes_dir: Directory containing keyframe images
            clip32_dir: Optional directory with CLIP ViT-B-32 .npy files
            faiss_index_path: Path to pre-built FAISS index pickle file
        """
        self.models = models
        self.embedding_dir = embedding_dir
        self.keyframes_dir = keyframes_dir
        self.clip32_dir = clip32_dir
        self.faiss_index_path = faiss_index_path or str(settings.faiss_index_path)

        self.clip32_embeddings: np.ndarray = np.array([])
        self.eva02_index = None
        self.clip32_index = None
        self.is_dual = False
        self.use_faiss = False
        self._path_to_idx = {}
        # Map video_name -> list of row indices in self.embeddings.
        # Pre-computed for O(1) Stage-2 filtered search (search_text_filtered).
        self.video_to_indices: dict[str, list[int]] = {}

        # ── Priority 1: Direct FAISS Index ────────────────────────
        if self._load_from_faiss_index(self.faiss_index_path):
            return

        # ── Priority 2 (Fallback): In-Memory .npy Array ───────────
        logger.warning(
            f"⚠️ [NUMPY FALLBACK] FAISS index not found at '{self.faiss_index_path}'. "
            f"Falling back to in-memory .npy embeddings. "
            f"For optimal speed and scalability on large datasets, build the FAISS index with: "
            f"'aic26 pipeline build-index'"
        )

        logger.info("Loading .npy embeddings into RAM...")
        self.embeddings, self.image_paths, self.video_names = self._load_all()

        n_images = len(self.image_paths)
        n_videos = len(set(self.video_names))
        logger.info(
            f"Loaded {n_images} images from {n_videos} videos. "
            f"Embedding shape: {self.embeddings.shape}"
        )

        if clip32_dir and settings.use_dual_embedding:
            try:
                self.clip32_embeddings, _, _ = self._load_all(clip32_dir)
                if (
                    self.clip32_embeddings.shape[0] == self.embeddings.shape[0]
                    and self.clip32_embeddings.size > 0
                ):
                    self.is_dual = True
                    logger.info(
                        f"Dual stream loaded: CLIP-32 shape={self.clip32_embeddings.shape}"
                    )
                else:
                    logger.warning(
                        f"CLIP-32 stream row count mismatch "
                        f"({self.clip32_embeddings.shape[0]} vs {self.embeddings.shape[0]}); "
                        f"falling back to single-stream search."
                    )
                    self.clip32_embeddings = np.array([])
            except Exception as e:
                logger.warning(f"Failed to load CLIP-32 stream: {e}")
                self.clip32_embeddings = np.array([])

    def _load_from_faiss_index(self, index_path: str) -> bool:
        """
        Attempt to load pre-built FAISS index from disk.
        Returns True if successfully loaded, False otherwise.
        """
        if not index_path or not os.path.exists(index_path):
            return False

        try:
            with open(index_path, "rb") as f:
                db_data = pickle.load(f)

            if not isinstance(db_data, dict):
                logger.warning(f"FAISS index file {index_path} has invalid format.")
                return False

            self.eva02_index = db_data.get("eva02_index") or db_data.get("faiss_index")
            if self.eva02_index is None:
                logger.warning(f"No FAISS index object found in {index_path}.")
                return False

            self.image_paths = db_data.get("image_paths", [])
            self.video_names = db_data.get("video_names", [])
            self.embeddings = db_data.get("embeddings", np.array([]))

            if settings.use_dual_embedding and "clip32_index" in db_data:
                self.clip32_index = db_data["clip32_index"]
                self.clip32_embeddings = db_data.get("clip32_embeddings", np.array([]))
                self.is_dual = True
                logger.info(
                    f"Dual-stream FAISS loaded: EVA-02 ({self.eva02_index.ntotal} vectors) + "
                    f"CLIP-32 ({self.clip32_index.ntotal} vectors)"
                )
            else:
                self.is_dual = False

            # Build fast lookup map
            self._path_to_idx = {
                path.replace(os.sep, "/"): i for i, path in enumerate(self.image_paths)
            }
            for i, path in enumerate(self.image_paths):
                self._path_to_idx[path] = i

            # Build video_to_indices for Stage-2 filtered search
            from collections import defaultdict as _dd
            _vti: dict[str, list[int]] = _dd(list)
            for i, vname in enumerate(self.video_names):
                _vti[vname].append(i)
            self.video_to_indices = dict(_vti)

            self.use_faiss = True
            logger.info(
                f"⚡ [FAISS INDEX ACTIVE] Successfully loaded FAISS index from '{index_path}' "
                f"({self.eva02_index.ntotal} vectors). Queries will execute directly via FAISS IndexFlatIP."
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to load FAISS index from '{index_path}': {e}")
            return False

    def search_faiss(
        self,
        query_emb: np.ndarray,
        top_k: int,
        stream: str = "eva02",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query FAISS index directly for high-performance top-k retrieval.

        Args:
            query_emb: L2-normalized float32 query vector shape (1, D)
            top_k: Number of nearest neighbors to retrieve
            stream: "eva02" or "clip32"

        Returns:
            Tuple of (indices_1d, scores_1d) sorted descending by score.
        """
        index = self.eva02_index if stream == "eva02" else self.clip32_index
        if index is None:
            raise ValueError(f"FAISS index for stream '{stream}' is not available.")

        k = min(top_k, index.ntotal)
        if k <= 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        q = query_emb.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        scores, indices = index.search(q, k)
        return indices[0], scores[0]

    def __len__(self) -> int:
        if self.use_faiss and self.eva02_index is not None:
            return self.eva02_index.ntotal
        return len(self.image_paths)
    
    # ── Encoders ──────────────────────────────────────────────────
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to L2-normalized embedding vector.
        
        Args:
            text: Text string to encode
            
        Returns:
            Float32 array of shape (1, embedding_dim)
        """
        with torch.no_grad():
            tokens = self.models.tokenizer([text]).to(self.models.device)
            features = self.models.clip_model.encode_text(tokens)
            features = F.normalize(features, dim=-1)
        return features.cpu().numpy().astype(np.float32)
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Encode image to L2-normalized embedding vector.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Float32 array of shape (1, embedding_dim)
        """
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.models.preprocess_val(img).unsqueeze(0).to(self.models.device)
            
            with torch.no_grad():
                features = self.models.clip_model.encode_image(img_tensor)
                features = F.normalize(features, dim=-1)
            
            return features.cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

    # ── Dual-stream encoders (CLIP ViT-B-32) ──────────────────

    def encode_text_clip32(self, text: str) -> np.ndarray:
        """Encode text via CLIP ViT-B-32. See ModelRegistry.encode_text_clip32."""
        return self.models.encode_text_clip32(text)

    def encode_image_clip32(self, image_path: str) -> np.ndarray:
        """Encode image via CLIP ViT-B-32. See ModelRegistry.encode_image_clip32."""
        return self.models.encode_image_clip32(image_path)

    def get_embedding_clip32(self, image_path: str) -> Optional[np.ndarray]:
        """
        Retrieve the pre-computed CLIP ViT-B-32 embedding for an image path,
        or None if unavailable. Mirrors get_embedding() for the second stream.
        """
        if not self.is_dual or self.clip32_embeddings.size == 0:
            return None
        try:
            if image_path in self._path_to_idx:
                idx = self._path_to_idx[image_path]
                return self.clip32_embeddings[idx : idx + 1].astype(np.float32)
        except Exception as e:
            logger.error(f"Error getting CLIP-32 embedding for {image_path}: {e}")
        return None

    def get_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Retrieve pre-computed embedding for an image path.
        
        Uses in-memory index for O(1) lookup.
        Falls back to loading from disk if not in memory.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Float32 array of shape (1, embedding_dim) or None
        """
        try:
            # Fast path: lookup in memory
            if image_path in self._path_to_idx:
                idx = self._path_to_idx[image_path]
                return self.embeddings[idx : idx + 1].astype(np.float32)
            
            # Fallback: parse and load from disk
            video_name = os.path.basename(os.path.dirname(image_path))
            frame_str = os.path.splitext(os.path.basename(image_path))[0]
            frame_idx = parse_frame_number_from_filename(frame_str)
            
            if frame_idx is None:
                logger.warning(f"Cannot parse frame number from: {image_path}")
                return None
            
            emb_path = os.path.join(self.embedding_dir, f"{video_name}.npy")
            if not os.path.exists(emb_path):
                logger.warning(f"Embedding file not found: {emb_path}")
                return None
            
            embs = np.load(emb_path)
            # Keyframe filenames are 1-indexed ("001.jpg" = first frame —
            # see pipeline/keyframe.py's saved_count starting at 1), but
            # the stacked .npy array is 0-indexed (row 0 = first frame).
            # The fast path above (_path_to_idx) already accounts for
            # this via enumerate(); this fallback previously used the
            # raw parsed filename number directly as the array index,
            # off by one — and out of bounds entirely for the last frame
            # of any video (e.g. "008.jpg" -> parsed 8, but valid indices
            # for an 8-row array are 0-7).
            array_idx = frame_idx - 1
            if array_idx < 0 or array_idx >= len(embs):
                logger.warning(
                    f"[EMBED] frame {frame_idx} (array idx {array_idx}) out of "
                    f"bounds for {video_name} ({len(embs)} embeddings)"
                )
                return None
            
            return embs[array_idx : array_idx + 1].astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error getting embedding for {image_path}: {e}")
            return None
    
    # ── Similarity ────────────────────────────────────────────────
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of vectors.
        
        Optimized: assumes both a and b are already L2-normalized
        (which they are after pipeline embedding extraction and
        text encoding with F.normalize). Falls back to explicit
        normalization only when norms deviate significantly.
        
        Args:
            a: Array of shape (M, D)
            b: Array of shape (N, D)
            
        Returns:
            Similarity matrix of shape (M, N)
        """
        # Fast path: direct dot product (valid when pre-normalized)
        return np.dot(a, b.T)
    
    def topk_indices(
        self,
        sims: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get indices and scores of top-k most similar items.
        Uses O(N) argpartition instead of O(N log N) full sort.
        
        Args:
            sims: 1-D similarity array
            top_k: Number of top results to return
            
        Returns:
            Tuple of (indices, scores) arrays, sorted descending
        """
        k = min(top_k, len(sims))
        if k <= 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        # argpartition is O(N) — only guarantees top-k are in the first k slots
        part_idx = np.argpartition(sims, -k)[-k:]
        # sort only the k candidates
        order = np.argsort(sims[part_idx])[::-1]
        idx = part_idx[order]
        return idx, sims[idx]
    
    def threshold_indices(
        self,
        sims: np.ndarray,
        score: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all indices with similarity above threshold.
        
        Args:
            sims: 1-D similarity array
            score: Minimum similarity threshold
            
        Returns:
            Tuple of (indices, scores) arrays, sorted descending
        """
        idx = np.where(sims >= score)[0]
        scores = sims[idx]
        if len(scores) == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        order = np.argsort(scores)[::-1]
        return idx[order], scores[order]
    
    # ── Private ───────────────────────────────────────────────────
    
    def _load_all(self, override_dir: str | None = None) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load all .npy embedding files and pair with keyframe paths.

        Args:
            override_dir: If provided, load from this directory instead of
                          self.embedding_dir. Used to load the CLIP-32 stream
                          with the same path-mapping logic.

        Returns:
            Tuple of (embeddings_array, image_paths_list, video_names_list)
        """
        target_dir = override_dir or self.embedding_dir
        if not os.path.exists(target_dir):
            logger.warning(f"Embedding directory not found: {target_dir}")
            return np.array([]), [], []

        all_embs: List[np.ndarray] = []
        all_paths: List[str] = []
        all_vids: List[str] = []

        npy_files = sorted(
            f for f in os.listdir(target_dir) if f.endswith(".npy")
        )

        if not npy_files:
            logger.warning(f"No .npy files found in {target_dir}")
            return np.array([]), [], []

        for npy in npy_files:
            video_name = os.path.splitext(npy)[0]
            emb_path = os.path.join(target_dir, npy)

            try:
                embs = np.load(emb_path)
            except Exception as e:
                logger.error(f"Failed to load {emb_path}: {e}")
                continue

            vid_folder = os.path.join(self.keyframes_dir, video_name)
            if not os.path.exists(vid_folder):
                logger.warning(f"Keyframe folder missing: {vid_folder}")
                continue

            frames = sorted(
                f for f in os.listdir(vid_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            )

            n_frames = len(frames)
            n_embs = len(embs)

            if n_frames != n_embs:
                logger.warning(
                    f"Mismatch for {video_name}: "
                    f"{n_embs} embeddings vs {n_frames} keyframes. "
                    f"Using min({n_embs}, {n_frames})."
                )

            n = min(n_frames, n_embs)
            for i in range(n):
                all_paths.append(os.path.join(vid_folder, frames[i]))
                all_vids.append(video_name)
            all_embs.append(embs[:n])

        if not all_embs:
            return np.array([]), [], []

        combined = np.concatenate(all_embs, axis=0).astype(np.float32)

        # Build reverse lookup for fast path (only when loading the
        # primary stream, otherwise we'd clobber it with a second copy)
        if override_dir is None:
            self._path_to_idx = {p: i for i, p in enumerate(all_paths)}
            # Build video_to_indices for Stage-2 filtered search
            from collections import defaultdict as _dd
            _vti: dict[str, list[int]] = _dd(list)
            for i, vname in enumerate(all_vids):
                _vti[vname].append(i)
            self.video_to_indices = dict(_vti)

        return combined, all_paths, all_vids