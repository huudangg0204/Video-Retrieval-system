"""
Data processing pipeline for AIC-26.

Provides modules for:
- Keyframe extraction from videos
- Object detection on keyframes
- OCR text extraction
- Embedding extraction (EVA-02)
- FAISS index building
- Pipeline orchestration

Usage:
    from aic26.pipeline import PipelineOrchestrator
    
    orchestrator = PipelineOrchestrator(settings)
    orchestrator.run(video_dir="./videos")
"""

from aic26.pipeline.keyframe import KeyframeExtractor
from aic26.pipeline.detection import ObjectDetector
from aic26.pipeline.ocr import OCRExtractor
from aic26.pipeline.embedding import EmbeddingExtractor
from aic26.pipeline.indexer import IndexBuilder
from aic26.pipeline.orchestrator import PipelineOrchestrator

__all__ = [
    "KeyframeExtractor",
    "ObjectDetector", 
    "OCRExtractor",
    "EmbeddingExtractor",
    "IndexBuilder",
    "PipelineOrchestrator",
]