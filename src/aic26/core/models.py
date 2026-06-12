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
        
        logger.info("All models loaded successfully.")
    
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
