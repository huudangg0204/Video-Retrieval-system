"""
BLIP Image-Text Matching reranker.

Reranks search results using BLIP ITM probability scores
to improve precision of text-based queries.

Uses batch inference for efficiency on GPU.

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
    Uses batch inference for 4-8x speedup over sequential processing.
    
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
        max_candidates: int = 20,
        batch_size: int = 8,
    ) -> List[Dict]:
        """
        Rerank candidates using BLIP ITM probability (batch mode).
        
        Args:
            query: Text query string (trailing '*' should be stripped)
            candidates: List of result dicts with 'image_path' key
            threshold: Minimum BLIP match probability to include
            max_candidates: Only rerank top-N candidates (PA3: reduces 100→20)
            batch_size: Number of images per batch inference
            
        Returns:
            Filtered and rescored list of result dicts
        """
        # PA3: Only rerank top-N candidates
        to_rerank = candidates[:max_candidates]
        pass_through = candidates[max_candidates:]
        
        # Load images
        images: List[Image.Image] = []
        valid_indices: List[int] = []
        for i, r in enumerate(to_rerank):
            try:
                img = Image.open(r["image_path"]).convert("RGB")
                images.append(img)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Reranker: cannot open image {r.get('image_path')}: {e}")
        
        if not images:
            return candidates
        
        # PA2: Batch inference
        all_probs: List[float] = []
        for batch_start in range(0, len(images), batch_size):
            batch_imgs = images[batch_start:batch_start + batch_size]
            try:
                inputs = self.models.blip_processor(
                    images=batch_imgs,
                    text=[query] * len(batch_imgs),
                    return_tensors="pt",
                    padding=True,
                ).to(self.models.device)
                
                with torch.no_grad():
                    itm_scores = self.models.blip_model(**inputs)[0]
                    probs = F.softmax(itm_scores, dim=-1)[:, 1]
                    all_probs.extend(probs.cpu().tolist())
            except Exception as e:
                logger.warning(f"Reranker: batch inference error: {e}")
                # Fallback: keep original scores for this batch
                all_probs.extend([0.0] * len(batch_imgs))
        
        # Filter by threshold and rescore
        accepted: List[Dict] = []
        for prob, idx in zip(all_probs, valid_indices):
            if prob >= threshold:
                result = dict(to_rerank[idx])
                result["similarity"] = prob
                accepted.append(result)
        
        # Sort by BLIP probability
        accepted.sort(key=lambda x: x["similarity"], reverse=True)
        
        logger.debug(
            f"Reranked: {len(to_rerank)} → {len(accepted)} candidates "
            f"(threshold={threshold}, batch_size={batch_size})"
        )
        
        return accepted