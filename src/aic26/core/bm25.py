from __future__ import annotations

import string
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from loguru import logger
from rank_bm25 import BM25Okapi

from aic26.config import settings
from aic26.pipeline.metadata import KeyframeMetadata


class BM25CaptionIndex:
    """
    BM25 text retrieval for keyframe captions.
    Aligned 1:1 with the main embedding index.
    """

    def __init__(self, image_paths: List[str], metadata_dir: str | Path | None = None) -> None:
        self.metadata_dir = Path(metadata_dir or settings.metadata_dir)
        self.image_paths = image_paths
        self.corpus: List[str] = []
        self.bm25: BM25Okapi | None = None

        self._load_and_build()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, strip punctuation, split on whitespace."""
        if not text:
            return []
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    def _load_and_build(self) -> None:
        if not self.metadata_dir.exists():
            logger.warning(f"Metadata directory not found: {self.metadata_dir}")
            return

        logger.info(f"Loading metadata for BM25 index ({len(self.image_paths)} items)...")
        tokenized_corpus = []

        # Load caption for each image_path
        for img_path_str in self.image_paths:
            img_path = Path(img_path_str)
            video_name = img_path.parent.name
            frame_idx = img_path.stem
            
            json_file = self.metadata_dir / video_name / f"{frame_idx}.json"
            cap = ""
            if json_file.exists():
                try:
                    meta = KeyframeMetadata.load(json_file)
                    cap = meta.caption or ""
                except Exception as e:
                    logger.debug(f"Error loading {json_file}: {e}")
            
            self.corpus.append(cap)
            tokenized_corpus.append(self._tokenize(cap))

        if any(len(c) > 0 for c in tokenized_corpus):
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("Built BM25 index over captions.")
        else:
            logger.warning("No valid captions found for BM25 index.")

    def get_scores(self, query: str) -> np.ndarray:
        """
        Get BM25 scores for a query, normalized to [0, 1] range via Min-Max scaling
        so it can be blended with cosine similarities.
        Returns array of shape (N,).
        """
        if not self.bm25:
            return np.zeros(len(self.image_paths), dtype=np.float32)

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return np.zeros(len(self.image_paths), dtype=np.float32)

        scores = np.array(self.bm25.get_scores(tokenized_query), dtype=np.float32)
        
        # Min-max scaling for blending
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores = (scores - s_min) / (s_max - s_min)
        else:
            scores = np.zeros_like(scores)

        return scores
