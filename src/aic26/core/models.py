"""
Lazy-loaded model registry for AIC-26.

Centralizes all neural network model loading to ensure:
- Single initialization (avoids duplicate loading bugs)
- GPU/CPU device management
- Consistent interfaces
- Lazy loading of heavy models to save VRAM

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
    - EVA-02 CLIP (text + image encoder) — eager loaded
    - CLIP ViT-B-32 (dual-stream) — lazy loaded on first use
    - Faster R-CNN (object detector) — lazy loaded on first use
    - BLIP ITM (image-text matching reranker) — lazy loaded on first use
    
    Usage:
        models = ModelRegistry()
        text_emb = models.clip_model.encode_text(tokens)
    """
    
    def __init__(self) -> None:
        """Initialize core models on the appropriate device."""
        requested = (settings.device or "cuda").lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                f"[DEVICE] settings.device={requested!r} but CUDA unavailable — using cpu "
                f"(set DEVICE=cpu in .env to silence this)"
            )
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(requested)
        logger.info(f"Using device: {self.device}")
        
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Eager: always needed for search
        self._load_clip()

        # Lazy: only loaded on first access (saves VRAM)
        self._detector = None
        self._detection_transform = None
        self._blip_processor = None
        self._blip_model = None
        self.clip32_model = None
        self.clip32_processor = None

        logger.info("Primary retrieval model loaded (EVA-02 CLIP). Auxiliary models (Detector/BLIP/CLIP-32) are lazy-loaded.")
    
    # ── CLIP / EVA-02 (eager) ─────────────────────────────────────
    
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
    
    # ── Object detector (lazy) ────────────────────────────────────
    
    @property
    def detector(self):
        """Lazy-load Faster R-CNN on first access."""
        if self._detector is None:
            self._load_detector()
        return self._detector

    @property
    def detection_transform(self):
        """Get detection transform, loading detector if needed."""
        if self._detection_transform is None:
            self._load_detector()
        return self._detection_transform

    def _load_detector(self) -> None:
        """Load Faster R-CNN for object detection."""
        logger.info("Lazy-loading Faster R-CNN detector...")
        
        self._detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        )
        self._detector.to(self.device).eval()
        self._detection_transform = Compose([ToTensor()])
        
        logger.info("Faster R-CNN ready.")
    
    # ── BLIP reranker (lazy) ──────────────────────────────────────
    
    @property
    def blip_processor(self):
        """Lazy-load BLIP processor on first access."""
        if self._blip_processor is None:
            self._load_reranker()
        return self._blip_processor

    @property
    def blip_model(self):
        """Lazy-load BLIP ITM model on first access."""
        if self._blip_model is None:
            self._load_reranker()
        return self._blip_model

    def _load_reranker(self) -> None:
        """Load BLIP Image-Text Matching model for reranking."""
        logger.info(f"Lazy-loading BLIP ITM: {settings.blip_model_name}")
        
        self._blip_processor = BlipProcessor.from_pretrained(settings.blip_model_name)
        self._blip_model = BlipForImageTextRetrieval.from_pretrained(
            settings.blip_model_name
        ).to(self.device)

        logger.info("BLIP ITM ready.")

    # ── CLIP ViT-B-32 for dual-embedding pipeline (lazy) ─────────

    def _load_clip32(self) -> None:
        """
        Lazy-load CLIP ViT-B-32 model and processor for dual-embedding.

        Unlike BLIP-2 Q-Former (image-only), CLIP ViT-B-32 natively encodes
        both text and images into a shared 512-d embedding space. This
        eliminates the black-image hack previously required for text queries.
        """
        from transformers import CLIPModel, CLIPProcessor

        if getattr(self, "clip32_model", None) is not None:
            return

        logger.info(f"Loading CLIP ViT-B-32: {settings.clip32_model_name}")
        self.clip32_processor = CLIPProcessor.from_pretrained(
            settings.clip32_model_name
        )
        self.clip32_model = (
            CLIPModel.from_pretrained(settings.clip32_model_name)
            .to(self.device)
            .eval()
        )
        self.clip32_embed_dim = int(self.clip32_model.config.projection_dim)
        logger.info(
            f"CLIP ViT-B-32 ready. Projection dim: {self.clip32_embed_dim}"
        )

    def encode_text_clip32(self, text: str) -> "np.ndarray":
        """
        Encode text via CLIP ViT-B-32 natively.

        Returns:
            Float32 array of shape (1, 512) — L2-normalized.
        """
        import numpy as np

        if getattr(self, "clip32_model", None) is None:
            self._load_clip32()

        # Use tokenizer directly (avoids forward() requiring pixel_values
        # in newer transformers versions).
        text_inputs = self.clip32_processor.tokenizer(
            [text], return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)
        with torch.no_grad():
            text_outputs = self.clip32_model.text_model(**text_inputs)
            pooled = text_outputs.pooler_output
            feats = self.clip32_model.text_projection(pooled)
            feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy().astype(np.float32)

    def encode_image_clip32(self, image_path: str) -> "np.ndarray":
        """
        Encode a single image via CLIP ViT-B-32.

        Returns:
            Float32 array of shape (1, 512) — L2-normalized.
        """
        import numpy as np
        from PIL import Image

        if getattr(self, "clip32_model", None) is None:
            self._load_clip32()

        image = Image.open(image_path).convert("RGB")
        inputs = self.clip32_processor(
            images=image, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            image_outputs = self.clip32_model.vision_model(**inputs)
            pooled = image_outputs.pooler_output
            feats = self.clip32_model.visual_projection(pooled)
            feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy().astype(np.float32)

    def encode_images_clip32_batch(self, image_paths: list[str]) -> "np.ndarray":
        """
        Batch-encode images via CLIP ViT-B-32.

        Returns:
            Float32 array of shape (B, 512) — L2-normalized.
        """
        import numpy as np
        from PIL import Image

        if getattr(self, "clip32_model", None) is None:
            self._load_clip32()

        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.clip32_processor(
            images=images, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            image_outputs = self.clip32_model.vision_model(**inputs)
            pooled = image_outputs.pooler_output
            feats = self.clip32_model.visual_projection(pooled)
            feats = F.normalize(feats, dim=-1)
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