"""
Core engine modules for AIC-26.

Contains the main retrieval engine components.
"""

from aic26.core.models import ModelRegistry
from aic26.core.embeddings import EmbeddingIndex
from aic26.core.database import VectorDatabase
from aic26.core.search import SearchEngine
from aic26.core.rerank import Reranker
from aic26.core.trake import TrakeEngine
from aic26.core.export import Exporter
from aic26.core.retrieval import EVA02ImageRetrieval
from aic26.core.ocr_search import OCRSearchEngine, get_ocr_search_engine

__all__ = [
    "ModelRegistry",
    "EmbeddingIndex",
    "VectorDatabase",
    "SearchEngine",
    "Reranker",
    "TrakeEngine",
    "Exporter",
    "EVA02ImageRetrieval",
    "OCRSearchEngine",
    "get_ocr_search_engine",
]