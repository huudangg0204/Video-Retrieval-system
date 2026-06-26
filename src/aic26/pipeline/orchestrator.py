"""
Pipeline orchestrator for AIC-26.

Coordinates the complete data processing pipeline:
1. Keyframe extraction
2. Object detection
3. OCR text extraction
4. Embedding extraction
5. FAISS index building

Migrated from: (new module - orchestrates previous scripts)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

from loguru import logger


from aic26.config.settings import Settings

from aic26.pipeline.keyframe import KeyframeExtractor
from aic26.pipeline.detection import ObjectDetector
from aic26.pipeline.ocr import OCRExtractor
from aic26.pipeline.embedding import EmbeddingExtractor
from aic26.pipeline.indexer import IndexBuilder


class PipelineOrchestrator:
    """
    Orchestrates the complete data processing pipeline.
    
    Runs all steps in sequence with proper error handling,
    progress tracking, and timing information.
    
    Usage:
        orchestrator = PipelineOrchestrator(settings)
        results = orchestrator.run(
            video_dir="./videos",
            skip_ocr=True,
        )
    """
    
    def __init__(self, config: Optional['Settings'] = None) -> None:
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config: Settings instance
        """
        if config is None:
            from aic26.config import settings as global_settings
            config = global_settings
        
        self.config = config
        self.steps_completed: list[str] = []
        self.steps_failed: list[str] = []
        self.steps_skipped: list[str] = []
        self.start_time: float = 0.0
    
    def run(
        self,
        video_dir: str,
        skip_detection: bool = False,
        skip_ocr: bool = False,
        skip_embeddings: bool = False,
        skip_index: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the complete data processing pipeline.
        
        Steps:
        1. Extract keyframes from videos
        2. Object detection on keyframes (optional)
        3. OCR text extraction (optional)
        4. Extract EVA-02 embeddings (optional)
        5. Build FAISS index (optional)
        
        Args:
            video_dir: Directory containing video files
            skip_detection: Skip object detection step
            skip_ocr: Skip OCR step
            skip_embeddings: Skip embedding extraction
            skip_index: Skip FAISS index building
            
        Returns:
            Dict with pipeline results and timing info
        """
        self.start_time = time.time()
        results: Dict[str, Any] = {}
        
        logger.info("=" * 60)
        logger.info("AIC-26 Data Pipeline Starting")
        logger.info("=" * 60)
        
        # ── Step 1: Keyframe Extraction ──────────────────────────
        logger.info("Step 1/5: Extracting keyframes from videos...")
        step_start = time.time()
        
        try:
            extractor = KeyframeExtractor(self.config)
            results['keyframes'] = extractor.process(video_dir)
            self.steps_completed.append("keyframe_extraction")
            
            duration = time.time() - step_start
            logger.info(f"✅ Keyframe extraction complete ({duration:.1f}s)")
            
        except Exception as e:
            logger.error(f"❌ Keyframe extraction failed: {e}")
            self.steps_failed.append("keyframe_extraction")
            raise
        
        # ── Step 2: Object Detection ─────────────────────────────
        if not skip_detection:
            logger.info("Step 2/5: Detecting objects in keyframes...")
            step_start = time.time()
            
            try:
                detector = ObjectDetector(self.config)
                results['detections'] = detector.process(
                    str(self.config.keyframes_dir),
                )
                self.steps_completed.append("object_detection")
                
                duration = time.time() - step_start
                logger.info(f"✅ Object detection complete ({duration:.1f}s)")
                
            except Exception as e:
                logger.warning(f"⚠️  Object detection failed (continuing): {e}")
                self.steps_failed.append("object_detection")
        else:
            logger.info("Step 2/5: Object detection SKIPPED")
            self.steps_skipped.append("object_detection")
        
        # ── Step 3: OCR Extraction ───────────────────────────────
        if not skip_ocr:
            logger.info("Step 3/5: Extracting text with OCR...")
            step_start = time.time()
            
            try:
                ocr = OCRExtractor(self.config)
                results['ocr'] = ocr.process(
                    input_dir=str(self.config.keyframes_dir),
                    num_workers=self.config.max_workers,
                    parallel_mode="image"
                )
                self.steps_completed.append("ocr_extraction")
                
                duration = time.time() - step_start
                logger.info(f"✅ OCR extraction complete ({duration:.1f}s)")
                
            except Exception as e:
                logger.warning(f"⚠️  OCR extraction failed (continuing): {e}")
                self.steps_failed.append("ocr_extraction")
        else:
            logger.info("Step 3/5: OCR extraction SKIPPED")
            self.steps_skipped.append("ocr_extraction")
        
        # ── Step 4: Embedding Extraction ─────────────────────────
        if not skip_embeddings:
            logger.info("Step 4/5: Extracting EVA-02 embeddings...")
            step_start = time.time()
            
            try:
                embedder = EmbeddingExtractor(self.config)
                results['embeddings'] = embedder.process_images(
                    str(self.config.keyframes_dir),
                )
                self.steps_completed.append("embedding_extraction")
                
                duration = time.time() - step_start
                logger.info(f"✅ Embedding extraction complete ({duration:.1f}s)")
                
            except Exception as e:
                logger.error(f"❌ Embedding extraction failed: {e}")
                self.steps_failed.append("embedding_extraction")
                raise
        else:
            logger.info("Step 4/5: Embedding extraction SKIPPED")
            self.steps_skipped.append("embedding_extraction")
        
                # ── Step 4b: Caption Embeddings (if caption data exists) ─
        if not skip_embeddings:
            caption_dir = os.path.join(
                str(self.config.data_dir), "blip_index", "checkpoints"
            )
            if os.path.exists(caption_dir):
                logger.info("Step 4b/5: Extracting caption embeddings...")
                step_start = time.time()
                
                try:
                    embedder = EmbeddingExtractor(self.config)
                    caption_output = str(self.config.embedding_dir).replace(
                        "embeddings", "caption_embeddings"
                    )
                    results['caption_embeddings'] = embedder.process_captions(
                        input_dir=caption_dir,
                        output_dir=caption_output,
                    )
                    self.steps_completed.append("caption_embedding_extraction")
                    
                    duration = time.time() - step_start
                    logger.info(f"✅ Caption embeddings complete ({duration:.1f}s)")
                    
                except Exception as e:
                    logger.warning(f"⚠️  Caption embedding failed (continuing): {e}")
                    self.steps_failed.append("caption_embedding_extraction")
            else:
                logger.info(f"Step 4b/5: Caption data not found at {caption_dir}, skipping")
                self.steps_skipped.append("caption_embedding_extraction")
                
        # ── Step 5: FAISS Index Building ─────────────────────────
        if not skip_index:
            logger.info("Step 5/5: Building FAISS index...")
            step_start = time.time()
            
            try:
                builder = IndexBuilder(self.config)
                results['index'] = builder.build(
                    str(self.config.embedding_dir),
                )
                self.steps_completed.append("index_building")
                
                duration = time.time() - step_start
                logger.info(f"✅ FAISS index built ({duration:.1f}s)")
                
            except Exception as e:
                logger.error(f"❌ Index building failed: {e}")
                self.steps_failed.append("index_building")
                raise
        else:
            logger.info("Step 5/5: FAISS index building SKIPPED")
            self.steps_skipped.append("index_building")
        
        # ── Summary ──────────────────────────────────────────────
        total_duration = time.time() - self.start_time
        
        logger.info("=" * 60)
        logger.info("Pipeline Summary")
        logger.info(f"  Completed: {len(self.steps_completed)} steps → {self.steps_completed}")
        logger.info(f"  Failed:    {len(self.steps_failed)} steps → {self.steps_failed}")
        logger.info(f"  Skipped:   {len(self.steps_skipped)} steps → {self.steps_skipped}")
        logger.info(f"  Total time: {total_duration:.1f}s ({total_duration/60:.1f}min)")
        logger.info("=" * 60)
        
        results['_pipeline_stats'] = {
            'steps_completed': self.steps_completed,
            'steps_failed': self.steps_failed,
            'steps_skipped': self.steps_skipped,
            'total_duration_seconds': total_duration,
        }
        
        return results