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

import typing

if typing.TYPE_CHECKING:
    from aic26.pipeline.keyframe import KeyframeExtractor
    from aic26.pipeline.detection import ObjectDetector
    from aic26.pipeline.ocr import OCRExtractor
    from aic26.pipeline.ocr_gpu import OCRExtractorGPU
    from aic26.pipeline.embedding import EmbeddingExtractor
    from aic26.pipeline.indexer import IndexBuilder
    from aic26.pipeline.orchestrator import PipelineOrchestrator

__all__ = [
    "KeyframeExtractor",
    "ObjectDetector", 
    "OCRExtractor",
    "OCRExtractorGPU",
    "EmbeddingExtractor",
    "IndexBuilder",
    "PipelineOrchestrator",
]

def __getattr__(name: str):
    if name == "KeyframeExtractor":
        from aic26.pipeline.keyframe import KeyframeExtractor
        return KeyframeExtractor
    elif name == "ObjectDetector":
        from aic26.pipeline.detection import ObjectDetector
        return ObjectDetector
    elif name == "OCRExtractor":
        from aic26.pipeline.ocr import OCRExtractor
        return OCRExtractor
    elif name == "OCRExtractorGPU":
        from aic26.pipeline.ocr_gpu import OCRExtractorGPU
        return OCRExtractorGPU
    elif name == "EmbeddingExtractor":
        from aic26.pipeline.embedding import EmbeddingExtractor
        return EmbeddingExtractor
    elif name == "IndexBuilder":
        from aic26.pipeline.indexer import IndexBuilder
        return IndexBuilder
    elif name == "PipelineOrchestrator":
        from aic26.pipeline.orchestrator import PipelineOrchestrator
        return PipelineOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")