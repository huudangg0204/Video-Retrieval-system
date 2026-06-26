"""
Lazy-loaded model registry for AIC-26.

Centralizes all neural network model loading to ensure:
- Single initialization (avoids duplicate loading bugs)
- GPU/CPU device management
- Consistent interfaces

Migrated from: core/models.py
"""
from __future__ import annotations

import open_clip
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, ToTensor

from transformers import BlipProcessor, BlipForImageTextRetrieval

from loguru import logger

from aic26.config import settings


class ModelRegistry:
    """
    Loads and owns all neural-network models.
    
    Guarantees single-initialization of each model.
    Provides unified access to:
    - EVA-02 CLIP (text + image encoder)
    - Faster R-CNN (object detector)
    - BLIP ITM (image-text matching reranker)
    
    Usage:
        models = ModelRegistry()
        text_emb = models.clip_model.encode_text(tokens)
        objects = models.detector([image_tensor])
    """
    
    def __init__(self) -> None:
        """Initialize all models on the appropriate device."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self._load_clip()
        self._load_detector()
        self._load_reranker()

        # BLIP-2 is heavy (~1.2B params). Lazy-load on first encode_blip2
        # call to keep startup time and base VRAM low.
        self.blip2_model = None
        self.blip2_processor = None

        logger.info("All eager models loaded successfully.")
    
    # ── CLIP / EVA-02 ──────────────────────────────────────────────
    
    def _load_clip(self) -> None:
        """Load the EVA-02 CLIP model for text and image encoding."""
        logger.info(f"Loading EVA-02 CLIP: {settings.eva02_model_name}")
        
        self.clip_model, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms(
                settings.eva02_model_name,
                pretrained=settings.eva02_pretrained,
            )
        )
        self.tokenizer = open_clip.get_tokenizer(settings.eva02_model_name)
        self.clip_model = self.clip_model.to(self.device).eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.clip_model.parameters())
        logger.info(f"EVA-02 CLIP loaded. Parameters: {total_params / 1e6:.1f}M")
    
    # ── Object detector ───────────────────────────────────────────
    
    def _load_detector(self) -> None:
        """Load Faster R-CNN for object detection."""
        logger.info("Loading Faster R-CNN detector...")
        
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        )
        self.detector.to(self.device).eval()
        self.detection_transform = Compose([ToTensor()])
        
        logger.info("Faster R-CNN ready.")
    
    # ── BLIP reranker ─────────────────────────────────────────────
    
    def _load_reranker(self) -> None:
        """Load BLIP Image-Text Matching model for reranking."""
        logger.info(f"Loading BLIP ITM: {settings.blip_model_name}")
        
        self.blip_processor = BlipProcessor.from_pretrained(settings.blip_model_name)
        self.blip_model = BlipForImageTextRetrieval.from_pretrained(
            settings.blip_model_name
        ).to(self.device)

        logger.info("BLIP ITM ready.")

    # ── BLIP-2 (Q-Former) for dual-embedding pipeline ─────────────

    def _load_blip2(self) -> None:
        """
        Lazy-load BLIP-2 model and processor for dual-embedding extraction.

        BLIP-2 here is used as an image+text embedding source (Q-Former pooled
        query embeddings), NOT for text generation. We expose encode_text_blip2
        and encode_image_blip2 returning L2-normalized 768-d vectors compatible
        with cosine similarity.
        """
        from transformers import Blip2Model, Blip2Processor

        if getattr(self, "blip2_model", None) is not None:
            return

        logger.info(f"Loading BLIP-2 (Q-Former): {settings.blip2_model_name}")
        self.blip2_processor = Blip2Processor.from_pretrained(settings.blip2_model_name)
        self.blip2_model = (
            Blip2Model.from_pretrained(settings.blip2_model_name)
            .to(self.device)
            .eval()
        )
        # Q-Former hidden size lives in the nested config in transformers 5.x
        qcfg = getattr(self.blip2_model.config, "qformer_config", None)
        if qcfg is not None and getattr(qcfg, "hidden_size", None):
            self.blip2_embed_dim = int(qcfg.hidden_size)
        else:
            self.blip2_embed_dim = int(
                getattr(self.blip2_model.config, "image_text_hidden_size", 768) or 768
            )
        logger.info(
            f"BLIP-2 ready. Q-Former hidden size: {self.blip2_embed_dim}"
        )

    def _pool_blip2_query(self, query_out) -> "torch.Tensor":
        """
        Reduce a (batch, num_query_tokens, hidden) Q-Former output to a
        single vector per item. We use the first query token (CLS-style)
        followed by L2-normalize, matching the BLIP-2 retrieval recipe.
        """
        # Newer transformers returns Tensor directly; older returns ModelOutput-like
        if isinstance(query_out, torch.Tensor):
            feats = query_out[:, 0, :]
        else:
            hidden = getattr(query_out, "last_hidden_state", None)
            if hidden is None:
                hidden = query_out[0]
            feats = hidden[:, 0, :]
        return torch.nn.functional.normalize(feats, dim=-1)

    def encode_image_blip2(self, image_path: str) -> "np.ndarray":
        """
        Encode a single image via BLIP-2 Q-Former.

        Returns:
            Float32 array of shape (1, blip2_embed_dim) — L2-normalized.
        """
        import numpy as np
        from PIL import Image

        if getattr(self, "blip2_model", None) is None:
            self._load_blip2()

        image = Image.open(image_path).convert("RGB")
        inputs = self.blip2_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            query_out = self.blip2_model.get_qformer_features(**inputs)
        feats = self._pool_blip2_query(query_out)
        return feats.cpu().numpy().astype(np.float32)

    def encode_text_blip2(self, text: str, image_path: str | None = None) -> "np.ndarray":
        """
        Encode text via BLIP-2 Q-Former.

        BLIP-2's Q-Former is image-conditioned (`get_qformer_features` only
        accepts pixel_values). For pure text queries we pass a 224x224 black
        image so the Q-Former still produces a stable embedding. If a real
        image is available (combined text+image query) pass image_path.
        """
        import numpy as np
        from PIL import Image

        if getattr(self, "blip2_model", None) is None:
            self._load_blip2()

        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        inputs = self.blip2_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            query_out = self.blip2_model.get_qformer_features(**inputs)
        feats = self._pool_blip2_query(query_out)
        return feats.cpu().numpy().astype(np.float32)

    def encode_images_blip2_batch(self, image_paths: list[str]) -> "np.ndarray":
        """
        Batch-encode images via BLIP-2. Returns float32 array of shape (B, 768).
        """
        import numpy as np
        from PIL import Image

        if getattr(self, "blip2_model", None) is None:
            self._load_blip2()

        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.blip2_processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            query_out = self.blip2_model.get_qformer_features(**inputs)
        feats = self._pool_blip2_query(query_out)
        return feats.cpu().numpy().astype(np.float32)
    
    # ── Utility ──────────────────────────────────────────────────
    
    def get_device_info(self) -> dict:
        """Get information about the compute device."""
        info = {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
            })
        return info
