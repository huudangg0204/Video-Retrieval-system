"""core/rerank.py — BLIP ITM reranking.

Fixes vs. original:
- Threshold is now a parameter (was hardcoded as a local `Threshold = 0.4`).
- Errors per image are caught and logged without silently dropping them.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image

import config
from .models import ModelRegistry


class Reranker:
    """Wraps BLIP Image-Text Matching for post-retrieval reranking."""

    def __init__(self, models: ModelRegistry) -> None:
        self.models = models

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        threshold: float = config.RERANK_THRESHOLD,
    ) -> List[Dict]:
        """
        Filter *candidates* to those where BLIP ITM probability >= *threshold*.

        Each passing candidate has its 'similarity' field replaced with the BLIP
        match probability, so downstream code can sort by it uniformly.

        Args:
            query:      The text query string (trailing '*' should be stripped beforehand).
            candidates: List of result dicts with 'image_path' key.
            threshold:  Minimum BLIP match probability to include a result.

        Returns:
            Filtered and scored list of result dicts.
        """
        accepted: List[Dict] = []
        for r in candidates:
            try:
                raw_image = Image.open(r["image_path"]).convert("RGB")
            except Exception as e:
                print(f"⚠️ Reranker: cannot open image {r.get('image_path')}: {e}")
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
                print(f"⚠️ Reranker: inference error for {r.get('image_path')}: {e}")

        return accepted
