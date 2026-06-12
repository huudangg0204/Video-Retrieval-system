"""
AIC-26: AI Challenge Video Retrieval System.

An intelligent video retrieval system using EVA-02 CLIP embeddings,
FAISS vector search, and temporal reasoning (TRAKE).
"""

__version__ = "1.0.0"
__author__ = "AIC Team"
__license__ = "MIT"

from aic26.config.settings import Settings, settings

__all__ = [
    "__version__",
    "__author__", 
    "__license__",
    "Settings",
    "settings",
]