"""
Embedding encoder for AIC-26.

Provides text and image encoding using EVA-02 CLIP and BLIP-ITM models.
No longer maintains an in-memory index — that responsibility belongs to
Elasticsearch.

Usage:
    encoder = EmbeddingEncoder(models)
    eva02_vec = encoder.encode_text_eva02("a red car")        # (1, 768)
    blip_vec  = encoder.encode_text_blip("a red car")         # (1, 768)
    eva02_img = encoder.encode_image_eva02("path/to/img.jpg") # (1, 768)
    blip_img  = encoder.encode_image_blip("path/to/img.jpg")  # (1, 768)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger

from aic26.core.models import ModelRegistry


class EmbeddingEncoder:
    """
    Thin wrapper that encodes text and images to embedding vectors.

    Delegates all neural-network calls to ModelRegistry.
    Does NOT store or load any embeddings from disk.

    Usage:
        encoder = EmbeddingEncoder(models)
        eva_vec, blip_vec = encoder.encode_text_dual("sunset over mountains")
    """

    def __init__(self, models: ModelRegistry) -> None:
        """
        Args:
            models: Loaded ModelRegistry instance.
        """
        self.models = models
        logger.debug("EmbeddingEncoder initialized.")

    # ── EVA-02 CLIP ───────────────────────────────────────────────────────

    def encode_text_eva02(self, text: str) -> np.ndarray:
        """
        Encode text to EVA-02 CLIP embedding.

        Args:
            text: Input string.

        Returns:
            Float32 array of shape (1, 768), L2-normalised.
        """
        with torch.no_grad():
            tokens = self.models.tokenizer([text]).to(self.models.device)
            feat = self.models.clip_model.encode_text(tokens)
            feat = F.normalize(feat, dim=-1)
        return feat.cpu().numpy().astype(np.float32)

    def encode_image_eva02(self, image_path: str) -> np.ndarray:
        """
        Encode an image file to EVA-02 CLIP embedding.

        Args:
            image_path: Absolute or relative path to an RGB image.

        Returns:
            Float32 array of shape (1, 768), L2-normalised.

        Raises:
            Exception: If the image cannot be opened or encoded.
        """
        img = Image.open(image_path).convert("RGB")
        tensor = self.models.preprocess_val(img).unsqueeze(0).to(self.models.device)
        with torch.no_grad():
            feat = self.models.clip_model.encode_image(tensor)
            feat = F.normalize(feat, dim=-1)
        return feat.cpu().numpy().astype(np.float32)

    # ── BLIP-ITM (Dual Embedding, Option C Lite) ──────────────────────────

    def encode_text_blip(self, text: str) -> np.ndarray:
        """
        Encode text to BLIP-ITM embedding.

        Args:
            text: Input query string.

        Returns:
            Float32 array of shape (1, 768), L2-normalised.
        """
        feat = self.models.encode_text_blip(text)
        return feat.cpu().numpy().astype(np.float32)

    def encode_image_blip(self, image_path: str) -> np.ndarray:
        """
        Encode an image file to BLIP-ITM embedding.

        Args:
            image_path: Absolute or relative path to an RGB image.

        Returns:
            Float32 array of shape (1, 768), L2-normalised.
        """
        img = Image.open(image_path).convert("RGB")
        feat = self.models.encode_image_blip(img)
        return feat.cpu().numpy().astype(np.float32)

    # ── Dual Convenience ──────────────────────────────────────────────────

    def encode_text_dual(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode text with both EVA-02 and BLIP models.

        Returns:
            Tuple of (eva02_vec, blip_vec), each shape (1, 768).
        """
        return self.encode_text_eva02(text), self.encode_text_blip(text)

    def encode_image_dual(
        self,
        image_path: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode an image with both EVA-02 and BLIP models.

        Returns:
            Tuple of (eva02_vec, blip_vec), each shape (1, 768).
        """
        img = Image.open(image_path).convert("RGB")

        # EVA-02
        tensor = self.models.preprocess_val(img).unsqueeze(0).to(self.models.device)
        with torch.no_grad():
            eva_feat = self.models.clip_model.encode_image(tensor)
            eva_feat = F.normalize(eva_feat, dim=-1)
        eva02_vec = eva_feat.cpu().numpy().astype(np.float32)

        # BLIP-ITM
        blip_feat = self.models.encode_image_blip(img)
        blip_vec = blip_feat.cpu().numpy().astype(np.float32)

        return eva02_vec, blip_vec


# ── Backward-compatibility shim ───────────────────────────────────────────────
# Keep EmbeddingIndex importable so existing imports don't break
# during the migration. Wraps EmbeddingEncoder with minimal stubs.

class EmbeddingIndex(EmbeddingEncoder):
    """
    Backward-compatibility shim for code that imports EmbeddingIndex.

    In the Elasticsearch architecture, full embedding loading is no longer
    needed — Elasticsearch stores and searches vectors.
    This class forwards encode_text / encode_image to EmbeddingEncoder.
    """

    def __init__(
        self,
        models: ModelRegistry,
        embedding_dir: Optional[str] = None,   # ignored — kept for API compat
        keyframes_dir: Optional[str] = None,   # ignored — kept for API compat
    ) -> None:
        super().__init__(models)
        self.embedding_dir = embedding_dir or ""
        self.keyframes_dir = keyframes_dir or ""
        # Stub attributes that old code may access
        self.embeddings: np.ndarray = np.array([])
        self.image_paths: list = []
        self.video_names: list = []
        logger.warning(
            "EmbeddingIndex is a legacy shim. "
            "Use EmbeddingEncoder for new code."
        )

    # Forward old method names to new ones
    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_text_eva02(text)

    def encode_image(self, image_path: str) -> np.ndarray:
        return self.encode_image_eva02(image_path)

    def get_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Stub — always returns None in ES architecture."""
        logger.debug(f"get_embedding called for {image_path} (ES mode: noop)")
        return None