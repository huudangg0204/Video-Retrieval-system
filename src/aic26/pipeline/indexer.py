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
    
    def build(
        self,
        embedding_dir: Optional[str] = None,
        output_path: Optional[str] = None,
        index_type: str = "flat_ip",
    ) -> dict:
        """
        Build FAISS index from embedding files.
        
        Args:
            embedding_dir: Directory containing .npy embedding files
            output_path: Path to save the FAISS database
            index_type: Type of FAISS index ('flat_ip' or 'flat_l2')
            
        Returns:
            Dict with build statistics
        """
        embedding_path = Path(embedding_dir) if embedding_dir else Path(self.config.embedding_dir)
        
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding directory not found: {embedding_path}")
        
        output_path = output_path or str(self.config.db_path)
        output_dir = os.path.dirname(output_path)
        ensure_dir(output_dir)
        
        logger.info(f"Loading embeddings from {embedding_path}...")
        embeddings = self._load_embeddings(str(embedding_path))
        
        if len(embeddings) == 0:
            logger.error("No embeddings loaded")
            return {"embeddings_loaded": 0, "index_size": 0}
        
        logger.info(f"Loaded {len(embeddings)} embeddings, dim={embeddings.shape[1]}")
        
        # Create FAISS index
        d = embeddings.shape[1]
        
        if index_type == "flat_ip":
            index = faiss.IndexFlatIP(d)
            logger.info(f"Created IndexFlatIP with dimension {d}")
        else:
            index = faiss.IndexFlatL2(d)
            logger.info(f"Created IndexFlatL2 with dimension {d}")
        
        # Add embeddings
        embeddings_f32 = embeddings.astype(np.float32)
        index.add(embeddings_f32)
        
        # Create database
        db_data = {
            'queries': [],
            'embeddings': embeddings_f32,
            'image_paths': [],
            'video_names': [],
            'frame_indices': [],
            'similarities': [],
            'timestamps': [],
            'metadata': {
                'build_date': np.datetime64('now').astype(str),
                'num_embeddings': len(embeddings),
                'dimension': d,
                'index_type': index_type,
            },
            'faiss_index': index,
        }
        
        # Save
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
        }
    
    def _load_embeddings(self, folder_path: str) -> np.ndarray:
        """
        Load all .npy embedding files from a folder.
        
        Args:
            folder_path: Path to folder containing .npy files
            
        Returns:
            Stacked numpy array of all embeddings
        """
        npy_files = sorted(
            f for f in os.listdir(folder_path)
            if f.endswith(".npy")
        )
        
        if not npy_files:
            logger.warning(f"No .npy files found in {folder_path}")
            return np.array([])
        
        all_embeddings = []
        
        for file in npy_files:
            file_path = os.path.join(folder_path, file)
            try:
                emb = np.load(file_path)
                if emb.ndim == 1:
                    emb = emb.reshape(1, -1)
                all_embeddings.append(emb)
                logger.debug(f"Loaded {file}: {emb.shape}")
            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")
        
        if not all_embeddings:
            return np.array([])
        
        return np.vstack(all_embeddings)