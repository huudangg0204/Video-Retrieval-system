"""
BLIP Image-Text Matching reranker.

Reranks search results using BLIP ITM probability scores
to improve precision of text-based queries.

Migrated from: core/rerank.py
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image

from loguru import logger

from aic26.config import settings
from aic26.core.models import ModelRegistry


class Reranker:
    """
    Reranks candidates using BLIP Image-Text Matching.
    
    Filters out results where BLIP ITM probability is below threshold.
    Replaces similarity scores with BLIP match probabilities.
    
    Usage:
        reranker = Reranker(models)
        reranked = reranker.rerank("a red car", candidates)
    """
    
    def __init__(self, models: ModelRegistry) -> None:
        """
        Initialize the reranker.
        
        Args:
            models: ModelRegistry with loaded BLIP model
        """
        self.models = models
        logger.debug("Reranker initialized with BLIP ITM")
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        threshold: float = settings.rerank_threshold,
    ) -> List[Dict]:
        """
        Rerank candidates using BLIP ITM probability.
        
        Args:
            query: Text query string (trailing '*' should be stripped)
            candidates: List of result dicts with 'image_path' key
            threshold: Minimum BLIP match probability to include
            
        Returns:
            Filtered and rescored list of result dicts
        """
        accepted: List[Dict] = []
        
        for i, r in enumerate(candidates):
            try:
                raw_image = Image.open(r["image_path"]).convert("RGB")
            except Exception as e:
                logger.warning(f"Reranker: cannot open image {r.get('image_path')}: {e}")
                continue
            
            try:
                inputs = self.models.blip_processor(
                    raw_image, query, return_tensors="pt"
                ).to(self.models.device)
                
                with torch.no_grad():
                    itm_score = self.models.blip_model(**inputs)[0]
                    prob = float(F.softmax(itm_score, dim=-1)[0, 1])
                
                if prob >= threshold:
                    result = dict(r)
                    result["similarity"] = prob
                    accepted.append(result)
                    
            except Exception as e:
                logger.warning(f"Reranker: inference error for {r.get('image_path')}: {e}")
        
        logger.debug(
            f"Reranked: {len(candidates)} → {len(accepted)} candidates "
            f"(threshold={threshold})"
        )
        
        return accepted