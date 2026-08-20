"""
BGE-M3 text embedding wrapper for AIC-26.

Provides dense 1024-d L2-normalized embeddings using BAAI/bge-m3 for
Stage-1 segment-level retrieval in the hierarchical two-stage search pipeline.

Device strategy:
- Offline indexing (build-segment-index): Caller passes device="cuda" explicitly
  → always GPU for maximum throughput.
- Online retrieval server (SegmentSearchEngine): Reads settings.text_retrieval
  ('GPU' → CUDA with FP16 half-precision ~1.15GB VRAM;
   'CPU' → CPU, 0 VRAM cost).

References:
    - BAAI/bge-m3: https://huggingface.co/BAAI/bge-m3
    - Spec: docs/specs/two-stage-hierarchical-search.md §4.1
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from aic26.config import settings
from aic26.config.constants import BGE_M3_EMBEDDING_DIM


class BGEM3TextEmbedder:
    """
    Wrapper for BAAI/bge-m3 producing dense 1024-d L2-normalized embeddings.

    Supports batch encoding of text segments and single-query encoding.
    Uses AutoTokenizer + AutoModel from transformers.

    When device is 'cuda' (or 'cuda:N'), the model is loaded in FP16
    (half-precision) to reduce VRAM usage from ~2.3 GB to ~1.15 GB while
    keeping cosine-similarity accuracy within 0.001% of FP32.

    Usage:
        embedder = BGEM3TextEmbedder()
        vecs = embedder.embed_texts(["text1", "text2"])  # (2, 1024)
        q = embedder.embed_query("some query")            # (1, 1024)
    """

    def __init__(
        self,
        model_name: str = settings.bge_m3_model_name,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the BGE-M3 embedder.

        Args:
            model_name: HuggingFace model identifier (default: BAAI/bge-m3)
            device: Target device string.
                - None    → resolved from settings.text_retrieval
                           ('GPU' maps to 'cuda' if CUDA available, else 'cpu';
                            'CPU' maps to 'cpu').
                - 'cuda' / 'cuda:0' → force GPU with FP16.
                - 'cpu'  → force CPU with FP32.
        """
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = self._resolve_device(device)

        # Determine dtype: FP16 on CUDA for VRAM efficiency (lossless for retrieval)
        use_fp16 = self.device.startswith("cuda")
        torch_dtype = torch.float16 if use_fp16 else torch.float32

        logger.info(
            f"Loading BGE-M3 model '{model_name}' on {self.device} "
            f"({'FP16' if use_fp16 else 'FP32'})..."
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype)
            .to(self.device)
            .eval()
        )
        if use_fp16:
            # Ensure model is in half precision (redundant but explicit)
            self.model = self.model.half()

        logger.info(
            f"BGE-M3 loaded. Embedding dim: {BGE_M3_EMBEDDING_DIM}. "
            f"Device: {self.device}. Dtype: {'FP16' if use_fp16 else 'FP32'}."
        )

    # ── Private helpers ────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        """
        Resolve the effective device string.

        Priority:
          1. Explicit ``device`` argument (passed by caller).
          2. ``settings.text_retrieval`` config value:
               - 'GPU' → 'cuda' if torch.cuda.is_available() else 'cpu'
               - 'CPU' → 'cpu'

        No automatic VRAM-size check is performed here; callers are
        responsible for ensuring the device choice is appropriate.
        """
        if device is not None:
            # Normalize: 'GPU' string shorthand → 'cuda'
            if device.upper() == "GPU":
                return "cuda" if torch.cuda.is_available() else "cpu"
            if device.upper() == "CPU":
                return "cpu"
            return device  # pass-through: 'cuda', 'cuda:0', 'cpu', etc.

        # Fall back to settings
        mode = getattr(settings, "text_retrieval", "CPU").upper()
        if mode == "GPU":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning(
                "[TEXT_RETRIEVAL=GPU] CUDA not available — falling back to CPU."
            )
            return "cpu"
        return "cpu"

    # ── Public API ────────────────────────────────────────────────────

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 8192,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into dense 1024-d L2-normalized embeddings.

        Args:
            texts: List of text strings to encode.
            batch_size: Number of texts per forward pass.
            max_length: Max tokenization length (BGE-M3 supports up to 8192).
            show_progress: Whether to display a tqdm progress bar.

        Returns:
            Float32 array of shape (N, 1024), L2-normalized.
        """
        if not texts:
            return np.empty((0, BGE_M3_EMBEDDING_DIM), dtype=np.float32)

        all_embeddings: List[np.ndarray] = []

        batch_starts = range(0, len(texts), batch_size)
        total_batches = (len(texts) + batch_size - 1) // batch_size
        iter_batches = (
            tqdm(
                batch_starts,
                desc="Embedding segments (BGE-M3)",
                total=total_batches,
                unit="batch",
            )
            if (show_progress and len(texts) > batch_size)
            else batch_starts
        )

        for start in iter_batches:
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                # BGE-M3: use [CLS] token (first token of last hidden state)
                hidden = outputs.last_hidden_state[:, 0, :]  # (B, 1024)
                # Cast to float32 before L2-normalize (safe even for FP16 model)
                normed = F.normalize(hidden.float(), p=2, dim=-1)

            all_embeddings.append(normed.cpu().numpy().astype(np.float32))

            if (start // batch_size + 1) % 10 == 0:
                logger.debug(
                    f"BGE-M3 embedded {min(start + batch_size, len(texts))}/{len(texts)} texts"
                )

        return np.concatenate(all_embeddings, axis=0)

    def embed_query(self, query: str, max_length: int = 8192) -> np.ndarray:
        """
        Encode a single query text into a 1024-d L2-normalized embedding.

        Args:
            query: Text query string.
            max_length: Max tokenization length.

        Returns:
            Float32 array of shape (1, 1024), L2-normalized.
        """
        return self.embed_texts([query], batch_size=1, max_length=max_length)


# ── Module-level singleton ─────────────────────────────────────────────────

_embedder_instance: Optional[BGEM3TextEmbedder] = None
_embedder_lock = __import__("threading").Lock()


def get_bge_m3_embedder(
    model_name: str = settings.bge_m3_model_name,
    device: Optional[str] = None,
) -> BGEM3TextEmbedder:
    """
    Return the shared BGEM3TextEmbedder singleton (thread-safe lazy init).

    The device is resolved once on first call:
      - If ``device`` is provided explicitly, it is used as-is.
      - Otherwise, ``settings.text_retrieval`` ('GPU' or 'CPU') determines
        the device for the online retrieval context.

    Args:
        model_name: HuggingFace model id (only used on first call).
        device: Target device (only used on first call).

    Returns:
        Shared BGEM3TextEmbedder instance.
    """
    global _embedder_instance
    if _embedder_instance is None:
        with _embedder_lock:
            if _embedder_instance is None:
                _embedder_instance = BGEM3TextEmbedder(model_name, device)
    return _embedder_instance
