"""core/embeddings.py — Embedding loading, FAISS indexing, and encoding.

Fixes vs. original:
- Validates that frame file count matches embedding row count per video (logs mismatch).
- get_embedding() uses the pre-loaded in-memory array instead of re-reading from disk.
- encode_text / encode_image are pure functions of the model, moved here from the monolith.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from utils.frame_utils import parse_frame_number_from_filename
from .models import ModelRegistry


class EmbeddingIndex:
    """
    Loads all .npy embedding files from *embedding_dir*, pairs them with their
    keyframe image paths, and exposes text/image encoding + cosine search helpers.
    """

    def __init__(self, models: ModelRegistry, embedding_dir: str, keyframes_dir: str) -> None:
        self.models        = models
        self.embedding_dir = embedding_dir
        self.keyframes_dir = keyframes_dir

        print("🔄 Loading embeddings…")
        self.embeddings, self.image_paths, self.video_names = self._load_all()
        print(f"✅ Loaded {len(self.image_paths)} images from {len(set(self.video_names))} videos.")

    # ── Encoders ──────────────────────────────────────────────────────────────

    def encode_text(self, text: str) -> np.ndarray:
        """Return L2-normalised text embedding, shape (1, D)."""
        with torch.no_grad():
            toks  = self.models.tokenizer([text]).to(self.models.device)
            feats = self.models.clip_model.encode_text(toks)
            feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy().astype(np.float32)

    def encode_image(self, image_path: str) -> np.ndarray:
        """Return L2-normalised image embedding, shape (1, D)."""
        img = Image.open(image_path).convert("RGB")
        img = self.models.preprocess_val(img).unsqueeze(0).to(self.models.device)
        with torch.no_grad():
            feats = self.models.clip_model.encode_image(img)
            feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy().astype(np.float32)

    def get_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Retrieve the pre-computed embedding for *image_path* from the in-memory index.

        Uses the already-loaded self.embeddings array rather than re-reading from disk,
        which is both faster and consistent with what FAISS is searching over.
        Returns shape (1, D) or None on error.
        """
        try:
            if image_path in self._path_to_idx:
                idx = self._path_to_idx[image_path]
                return self.embeddings[idx : idx + 1].astype(np.float32)

            # Fallback: parse video/frame from path and index into the numpy array
            video_name = os.path.basename(os.path.dirname(image_path))
            frame_str  = os.path.splitext(os.path.basename(image_path))[0]
            frame_idx  = parse_frame_number_from_filename(frame_str)

            if frame_idx is None:
                print(f"⚠️ Cannot parse frame number from: {image_path}")
                return None

            emb_path = os.path.join(self.embedding_dir, f"{video_name}.npy")
            if not os.path.exists(emb_path):
                print(f"⚠️ Embedding file not found: {emb_path}")
                return None

            embs = np.load(emb_path)
            if frame_idx >= len(embs):
                print(f"⚠️ Frame index {frame_idx} out of bounds for {video_name} ({len(embs)} embs)")
                return None
            return embs[frame_idx : frame_idx + 1].astype(np.float32)

        except Exception as e:
            print(f"❌ get_embedding error for {image_path}: {e}")
            return None

    # ── Similarity helpers ────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity between row-vectors in a and b. Returns (|a|, |b|) matrix."""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return np.dot(a_norm, b_norm.T)

    def topk_indices(self, sims: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) for the top-k entries of a 1-D similarity array."""
        idx = np.argsort(sims)[::-1][:top_k]
        return idx, sims[idx]

    def threshold_indices(
        self, sims: np.ndarray, score: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) for all entries with similarity >= score, sorted desc."""
        idx    = np.where(sims >= score)[0]
        scores = sims[idx]
        if len(scores) == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        order  = np.argsort(scores)[::-1]
        return idx[order], scores[order]

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_all(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load all .npy files from embedding_dir and pair each row with the
        corresponding sorted keyframe path.

        Logs a warning when the number of embedding rows doesn't match the
        number of keyframe images found on disk (instead of silently mispairing).
        """
        if not os.path.exists(self.embedding_dir):
            print(f"⚠️ Embedding dir not found: {self.embedding_dir}")
            return np.array([]), [], []

        all_embs: List[np.ndarray] = []
        all_paths: List[str]       = []
        all_vids:  List[str]       = []

        npy_files = sorted(f for f in os.listdir(self.embedding_dir) if f.endswith(".npy"))

        for npy in npy_files:
            video_name = os.path.splitext(npy)[0]
            emb_path   = os.path.join(self.embedding_dir, npy)

            try:
                embs = np.load(emb_path)
            except Exception as e:
                print(f"⚠️ Failed to load {emb_path}: {e}")
                continue

            vid_folder = os.path.join(self.keyframes_dir, video_name)
            if not os.path.exists(vid_folder):
                print(f"⚠️ Keyframe folder missing for {video_name}: {vid_folder}")
                continue

            frames = sorted(
                f for f in os.listdir(vid_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )

            if len(frames) != len(embs):
                print(
                    f"⚠️ Mismatch for {video_name}: "
                    f"{len(embs)} embeddings vs {len(frames)} keyframe files. "
                    f"Using min({len(embs)}, {len(frames)})."
                )

            n = min(len(frames), len(embs))
            for i in range(n):
                all_paths.append(os.path.join(vid_folder, frames[i]))
                all_vids.append(video_name)
            all_embs.append(embs[:n])

        if not all_embs:
            return np.array([]), [], []

        combined = np.concatenate(all_embs, axis=0).astype(np.float32)
        # Build reverse lookup: image_path → row index in combined
        self._path_to_idx = {p: i for i, p in enumerate(all_paths)}
        return combined, all_paths, all_vids
