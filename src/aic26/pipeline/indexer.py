"""
FAISS index builder for AIC-26.

Builds FAISS vector index from pre-computed embeddings
for efficient similarity search.

Migrated from: get_faiss.py
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from loguru import logger

from aic26.config import Settings
from aic26.utils.file_utils import ensure_dir


class IndexBuilder:
    """
    Build FAISS index from embedding files.
    
    Creates a FAISS IndexFlatIP (Inner Product) index from
    pre-computed EVA-02 embeddings saved as .npy files.
    
    Usage:
        builder = IndexBuilder(settings)
        builder.build("./embeddings", "./faiss_db.pkl")
    """
    
    def __init__(self, config: Optional['Settings'] = None) -> None:
        """
        Initialize the index builder.
        
        Args:
            config: Settings instance
        """
        if config is None:
            from aic26.config import settings as global_settings
            config = global_settings
        
        self.config = config
    
    def build_dual(
        self,
        eva02_dir: str,
        clip32_dir: str,
        output_path: Optional[str] = None,
        index_type: str = "flat_ip",
    ) -> dict:
        """
        Build a FAISS index pair from two embedding streams (EVA-02 + CLIP-32).

        Both directories are read with the same sorted video_name order so
        the row order of the two stacked arrays matches. The pickled DB
        contains both indexes plus the shared image_paths / video_names.

        The new pickle schema is:
            {
                "eva02_index": faiss.IndexFlatIP,
                "clip32_index": faiss.IndexFlatIP,
                "embeddings":  np.ndarray (eva02, kept for backward compat),
                "image_paths": list[str],
                "video_names": list[str],
                "weight_eva02": float,
                "weight_clip32": float,
                "metadata": {...},
            }

        Single-stream pickle (legacy "faiss_index" key) is still loadable;
        the loader in core/embeddings.py detects the missing clip32_index
        and falls back.
        """
        eva02_path = Path(eva02_dir)
        clip32_path = Path(clip32_dir)
        if not eva02_path.exists():
            raise FileNotFoundError(f"EVA-02 embedding dir not found: {eva02_path}")
        if not clip32_path.exists():
            raise FileNotFoundError(f"CLIP-32 embedding dir not found: {clip32_path}")

        output_path = output_path or str(self.config.faiss_index_path)
        ensure_dir(os.path.dirname(output_path))

        logger.info(f"Loading EVA-02 embeddings from {eva02_path}...")
        eva02_emb, image_paths, video_names = self._load_with_paths(str(eva02_path))
        logger.info(f"Loading CLIP-32 embeddings from {clip32_path}...")
        clip32_emb, _, _ = self._load_with_paths(str(clip32_path), image_paths, video_names)

        if eva02_emb.shape[0] != clip32_emb.shape[0]:
            raise ValueError(
                f"Stream row count mismatch: eva02={eva02_emb.shape[0]} clip32={clip32_emb.shape[0]}"
            )

        d_eva = eva02_emb.shape[1]
        d_clip = clip32_emb.shape[1]
        eva02_f32 = eva02_emb.astype(np.float32)
        clip32_f32 = clip32_emb.astype(np.float32)

        if index_type == "flat_ip":
            eva02_index = faiss.IndexFlatIP(d_eva)
            clip32_index = faiss.IndexFlatIP(d_clip)
        else:
            eva02_index = faiss.IndexFlatL2(d_eva)
            clip32_index = faiss.IndexFlatL2(d_clip)

        eva02_index.add(eva02_f32)
        clip32_index.add(clip32_f32)
        logger.info(
            f"FAISS dual index built: eva02={d_eva}-d, clip32={d_clip}-d, "
            f"n={eva02_index.ntotal}"
        )

        w_eva = float(getattr(self.config, "eva02_search_weight", 0.6))
        w_clip = float(getattr(self.config, "clip32_search_weight", 0.4))

        db_data = {
            "queries": [],
            # legacy single-stream key — keep so old loaders don't crash
            "embeddings": eva02_f32,
            "image_paths": image_paths,
            "video_names": video_names,
            "frame_indices": [],
            "similarities": [],
            "timestamps": [],
            "eva02_index": eva02_index,
            "clip32_index": clip32_index,
            "weight_eva02": w_eva,
            "weight_clip32": w_clip,
            "dual": True,
            "metadata": {
                "build_date": np.datetime64("now").astype(str),
                "num_embeddings": eva02_index.ntotal,
                "eva02_dim": d_eva,
                "clip32_dim": d_clip,
                "index_type": index_type,
                "weight_eva02": w_eva,
                "weight_clip32": w_clip,
            },
        }

        with open(output_path, "wb") as f:
            pickle.dump(db_data, f)

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(
            f"Dual FAISS saved to {output_path} "
            f"({file_size_mb:.1f} MB, {eva02_index.ntotal} vectors × 2 streams)"
        )
        return {
            "embeddings_loaded": eva02_index.ntotal,
            "eva02_dim": d_eva,
            "clip32_dim": d_clip,
            "index_type": index_type,
            "output_path": output_path,
            "file_size_mb": file_size_mb,
            "dual": True,
        }

    def _load_with_paths(
        self,
        folder_path: str,
        keyframes_dir: str | None = None,
        expected_paths: list[str] | None = None,
        expected_videos: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """
        Load all .npy files in a folder and stack them, pairing with actual keyframe paths.
        """
        kdir = keyframes_dir or str(self.config.keyframes_dir)
        npy_files = sorted(
            f for f in os.listdir(folder_path) if f.endswith(".npy")
        )
        if not npy_files:
            raise FileNotFoundError(f"No .npy files in {folder_path}")

        all_embs: list[np.ndarray] = []
        all_paths: list[str] = []
        all_vids: list[str] = []

        for npy in npy_files:
            video_name = os.path.splitext(npy)[0]
            file_path = os.path.join(folder_path, npy)
            try:
                emb = np.load(file_path)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)

            if expected_paths is not None:
                all_embs.append(emb)
                continue

            vid_folder = os.path.join(kdir, video_name)
            if os.path.exists(vid_folder):
                frames = sorted(
                    f for f in os.listdir(vid_folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                )
                n = min(len(frames), len(emb))
                for i in range(n):
                    all_paths.append(os.path.join(vid_folder, frames[i]).replace(os.sep, "/"))
                    all_vids.append(video_name)
                all_embs.append(emb[:n])
            else:
                n = len(emb)
                for i in range(n):
                    all_paths.append(f"{kdir}/{video_name}/{i:04d}.jpg")
                    all_vids.append(video_name)
                all_embs.append(emb)

        if not all_embs:
            return np.array([]), [], []

        combined = np.vstack(all_embs).astype(np.float32)

        if expected_paths is not None:
            if len(expected_paths) != combined.shape[0]:
                raise ValueError(
                    f"Stream length {combined.shape[0]} != expected {len(expected_paths)}"
                )
            return combined, expected_paths, expected_videos or []

        return combined, all_paths, all_vids

    def build_dual(
        self,
        eva02_dir: str,
        clip32_dir: str,
        output_path: Optional[str] = None,
        index_type: str = "flat_ip",
    ) -> dict:
        """
        Build a FAISS index pair from two embedding streams (EVA-02 + CLIP-32).
        """
        eva02_path = Path(eva02_dir)
        clip32_path = Path(clip32_dir)
        if not eva02_path.exists():
            raise FileNotFoundError(f"EVA-02 embedding dir not found: {eva02_path}")
        if not clip32_path.exists():
            raise FileNotFoundError(f"CLIP-32 embedding dir not found: {clip32_path}")

        output_path = output_path or str(self.config.faiss_index_path)
        ensure_dir(os.path.dirname(output_path))

        logger.info(f"Loading EVA-02 embeddings from {eva02_path}...")
        eva02_emb, image_paths, video_names = self._load_with_paths(str(eva02_path))
        logger.info(f"Loading CLIP-32 embeddings from {clip32_path}...")
        clip32_emb, _, _ = self._load_with_paths(
            str(clip32_path), expected_paths=image_paths, expected_videos=video_names
        )

        if eva02_emb.shape[0] != clip32_emb.shape[0]:
            raise ValueError(
                f"Stream row count mismatch: eva02={eva02_emb.shape[0]} clip32={clip32_emb.shape[0]}"
            )

        d_eva = eva02_emb.shape[1]
        d_clip = clip32_emb.shape[1]
        eva02_f32 = eva02_emb.astype(np.float32)
        clip32_f32 = clip32_emb.astype(np.float32)

        if index_type == "flat_ip":
            eva02_index = faiss.IndexFlatIP(d_eva)
            clip32_index = faiss.IndexFlatIP(d_clip)
        else:
            eva02_index = faiss.IndexFlatL2(d_eva)
            clip32_index = faiss.IndexFlatL2(d_clip)

        eva02_index.add(eva02_f32)
        clip32_index.add(clip32_f32)
        logger.info(
            f"FAISS dual index built: eva02={d_eva}-d, clip32={d_clip}-d, "
            f"n={eva02_index.ntotal}"
        )

        w_eva = float(getattr(self.config, "eva02_search_weight", 0.6))
        w_clip = float(getattr(self.config, "clip32_search_weight", 0.4))

        db_data = {
            "queries": [],
            "embeddings": eva02_f32,
            "clip32_embeddings": clip32_f32,
            "image_paths": image_paths,
            "video_names": video_names,
            "frame_indices": [],
            "similarities": [],
            "timestamps": [],
            "eva02_index": eva02_index,
            "clip32_index": clip32_index,
            "faiss_index": eva02_index,  # backwards compat
            "weight_eva02": w_eva,
            "weight_clip32": w_clip,
            "dual": True,
            "metadata": {
                "build_date": np.datetime64("now").astype(str),
                "num_embeddings": eva02_index.ntotal,
                "eva02_dim": d_eva,
                "clip32_dim": d_clip,
                "index_type": index_type,
                "weight_eva02": w_eva,
                "weight_clip32": w_clip,
            },
        }

        with open(output_path, "wb") as f:
            pickle.dump(db_data, f)

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(
            f"Dual FAISS saved to {output_path} "
            f"({file_size_mb:.1f} MB, {eva02_index.ntotal} vectors × 2 streams)"
        )
        return {
            "embeddings_loaded": eva02_index.ntotal,
            "eva02_dim": d_eva,
            "clip32_dim": d_clip,
            "index_type": index_type,
            "output_path": output_path,
            "file_size_mb": file_size_mb,
            "dual": True,
        }

    def build(
        self,
        embedding_dir: Optional[str] = None,
        output_path: Optional[str] = None,
        index_type: str = "flat_ip",
    ) -> dict:
        """
        Build FAISS index from embedding files.
        """
        embedding_path = Path(embedding_dir) if embedding_dir else Path(self.config.embedding_dir)
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding directory not found: {embedding_path}")

        output_path = output_path or str(self.config.faiss_index_path)
        ensure_dir(os.path.dirname(output_path))

        logger.info(f"Loading embeddings from {embedding_path}...")
        embeddings, image_paths, video_names = self._load_with_paths(str(embedding_path))

        if len(embeddings) == 0:
            logger.error("No embeddings loaded")
            return {"embeddings_loaded": 0, "index_size": 0}

        d = embeddings.shape[1]
        if index_type == "flat_ip":
            index = faiss.IndexFlatIP(d)
        else:
            index = faiss.IndexFlatL2(d)

        embeddings_f32 = embeddings.astype(np.float32)
        index.add(embeddings_f32)

        db_data = {
            "queries": [],
            "embeddings": embeddings_f32,
            "image_paths": image_paths,
            "video_names": video_names,
            "frame_indices": [],
            "similarities": [],
            "timestamps": [],
            "eva02_index": index,
            "faiss_index": index,
            "dual": False,
            "metadata": {
                "build_date": np.datetime64("now").astype(str),
                "num_embeddings": len(embeddings),
                "dimension": d,
                "index_type": index_type,
            },
        }

        with open(output_path, "wb") as f:
            pickle.dump(db_data, f)

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(
            f"FAISS index built and saved to {output_path} "
            f"({file_size_mb:.1f} MB, {len(embeddings)} vectors)"
        )

        return {
            "embeddings_loaded": len(embeddings),
            "dimension": d,
            "index_type": index_type,
            "output_path": output_path,
            "file_size_mb": file_size_mb,
            "dual": False,
        }