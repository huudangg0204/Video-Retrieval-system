"""
Configuration layer for AIC-26.

Provides type-safe settings via Pydantic and shared constants.
"""

from aic26.config.settings import Settings, settings
from aic26.config.constants import (
    COCO_CATEGORIES,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    EVA02_EMBEDDING_DIM,
)

__all__ = [
    "Settings",
    "settings",
    "COCO_CATEGORIES",
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "EVA02_EMBEDDING_DIM",
]
