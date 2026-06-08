"""core/models.py — Lazy-loaded model registry.

Fixes vs. original:
- Faster R-CNN loaded ONCE (was accidentally initialised twice in the constructor).
- Each model is a separate, named attribute for easy introspection.
- Centralises all heavy imports so the rest of the codebase stays lightweight.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import open_clip
from transformers import BlipProcessor, BlipForImageTextRetrieval

import config


class ModelRegistry:
    """
    Loads and owns all neural-network models.
    Guaranteed single-initialisation — avoids the duplicate Faster-RCNN bug.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")

        self._load_clip()
        self._load_detector()
        self._load_reranker()

    # ── CLIP / EVA02 ──────────────────────────────────────────────────────────

    def _load_clip(self) -> None:
        print("🔄 Loading EVA02 CLIP model…")
        self.clip_model, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms(
                config.MODEL_NAME, pretrained=config.MODEL_PRETRAINED
            )
        )
        self.tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)
        self.clip_model = self.clip_model.to(self.device).eval()
        print("✅ EVA02 CLIP ready.")

    # ── Object detector ───────────────────────────────────────────────────────

    def _load_detector(self) -> None:
        print("🔄 Loading Faster R-CNN detection model…")
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.detector.to(self.device).eval()
        self.detection_transform = T.Compose([T.ToTensor()])
        print("✅ Faster R-CNN ready.")

    # ── BLIP reranker ─────────────────────────────────────────────────────────

    def _load_reranker(self) -> None:
        print("🔄 Loading BLIP ITM reranker…")
        self.blip_processor = BlipProcessor.from_pretrained(config.BLIP_MODEL_NAME)
        self.blip_model     = BlipForImageTextRetrieval.from_pretrained(
            config.BLIP_MODEL_NAME
        ).to(self.device)
        print("✅ BLIP ITM ready.")
