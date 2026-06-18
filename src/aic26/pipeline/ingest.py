"""
Unified data ingestion pipeline for AIC-26.

Supports two input modes:
  Mode A — Keyframe folder (pre-extracted, e.g. Kaggle dataset format):
      input/
      ├── L01_V001/
      │   ├── 000.jpg
      │   └── 001.jpg
      └── L01_V002/

  Mode B — Raw video files:
      videos/
      ├── L01_V001.mp4
      └── L01_V002.mp4

Both modes converge into the same processing pipeline:
    Keyframes → [Detection] → [OCR] → EVA-02 Embedding → ES Index

Usage:
    pipeline = IngestPipeline()
    pipeline.run_from_keyframes("data/keyframes")   # Mode A
    pipeline.run_from_videos("data/videos")          # Mode B
"""
from __future__ import annotations

import os
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from tqdm import tqdm

from aic26.config import settings


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class IngestStats:
    """Statistics from a completed ingestion run."""
    input_mode: str = ""           # "keyframes" | "videos"
    input_path: str = ""
    keyframe_dir: str = ""
    videos_found: int = 0
    keyframes_found: int = 0
    embeddings_extracted: int = 0
    es_indexed: int = 0
    es_failed: int = 0
    steps_completed: List[str] = field(default_factory=list)
    steps_skipped: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"IngestStats(mode={self.input_mode}, "
            f"videos={self.videos_found}, "
            f"keyframes={self.keyframes_found}, "
            f"embedded={self.embeddings_extracted}, "
            f"indexed={self.es_indexed}, "
            f"duration={self.duration_seconds:.1f}s)"
        )


# ── Pipeline class ────────────────────────────────────────────────────────────

class IngestPipeline:
    """
    Unified ingestion pipeline: keyframes or videos → Elasticsearch.

    Orchestrates the full data preparation flow from raw input to
    searchable Elasticsearch index.

    Usage:
        pipeline = IngestPipeline()

        # From Kaggle keyframe dataset
        stats = pipeline.run_from_keyframes("path/to/keyframes/")

        # From raw video files
        stats = pipeline.run_from_videos("path/to/videos/")
    """

    def __init__(
        self,
        keyframes_dir: Optional[str] = None,
        embedding_dir: Optional[str] = None,
        detection_dir: Optional[str] = None,
        ocr_path: Optional[str] = None,
        batch_size: int = 40,
        skip_detection: bool = False,
        skip_ocr: bool = False,
        skip_es_index: bool = False,
        force_es_recreate: bool = False,
        encode_blip: bool = True,
        es_batch_size: int = 200,
    ) -> None:
        """
        Args:
            keyframes_dir: Where extracted keyframes are stored.
            embedding_dir: Where .npy embedding files are saved.
            detection_dir: Where detection JSONs are saved.
            ocr_path: Path to OCR results CSV.
            batch_size: GPU batch size for embedding extraction.
            skip_detection: Skip object detection step.
            skip_ocr: Skip OCR text extraction step.
            skip_es_index: Skip Elasticsearch indexing step.
            force_es_recreate: Delete and recreate the ES index.
            encode_blip: Encode BLIP-ITM embeddings during ES indexing.
            es_batch_size: Documents per ES bulk request.
        """
        self.keyframes_dir = keyframes_dir or str(settings.keyframes_dir)
        self.embedding_dir = embedding_dir or str(settings.embedding_dir)
        self.detection_dir = detection_dir or str(settings.detection_dir)
        self.ocr_path = ocr_path or str(settings.ocr_output_path)
        self.batch_size = batch_size
        self.skip_detection = skip_detection
        self.skip_ocr = skip_ocr
        self.skip_es_index = skip_es_index
        self.force_es_recreate = force_es_recreate
        self.encode_blip = encode_blip
        self.es_batch_size = es_batch_size

    # ── Public API ────────────────────────────────────────────────────────

    def run_from_keyframes(
        self,
        input_path: str,
        output_keyframes_dir: Optional[str] = None,
    ) -> IngestStats:
        """
        Mode A: Ingest from a pre-extracted keyframe directory or ZIP.

        Accepts:
        - A directory of subfolders (each subfolder = one video)
        - A .zip file with the same structure inside

        Args:
            input_path: Path to keyframe directory or .zip file.
            output_keyframes_dir: Where to copy/link keyframes if different
                                  from the default keyframes_dir.

        Returns:
            IngestStats with processing results.
        """
        stats = IngestStats(input_mode="keyframes", input_path=input_path)
        start = time.time()

        logger.info("=" * 60)
        logger.info("AIC-26 Ingest Pipeline — Mode A: Keyframes")
        logger.info(f"Input: {input_path}")
        logger.info("=" * 60)

        # ── Resolve keyframe directory ────────────────────────────────
        kf_dir = self._resolve_keyframe_input(
            input_path,
            output_keyframes_dir or self.keyframes_dir,
        )
        stats.keyframe_dir = kf_dir

        # Count structure
        videos = self._list_video_folders(kf_dir)
        stats.videos_found = len(videos)
        stats.keyframes_found = sum(
            len(self._list_images(os.path.join(kf_dir, v))) for v in videos
        )

        logger.info(
            f"Found {stats.videos_found} video folders, "
            f"{stats.keyframes_found} keyframes"
        )

        if stats.videos_found == 0:
            logger.error("No video subfolders found. Expected structure: input/<video_name>/<frame>.jpg")
            return stats

        # ── Run shared processing steps ───────────────────────────────
        self._run_processing_steps(kf_dir, stats)

        stats.duration_seconds = time.time() - start
        logger.info(f"Ingestion complete: {stats}")
        return stats

    def run_from_videos(
        self,
        video_dir: str,
    ) -> IngestStats:
        """
        Mode B: Ingest from raw video files.

        Extracts keyframes from each video, then runs the
        full processing pipeline.

        Args:
            video_dir: Directory containing .mp4 / .avi / .mkv files.

        Returns:
            IngestStats with processing results.
        """
        stats = IngestStats(input_mode="videos", input_path=video_dir)
        start = time.time()

        logger.info("=" * 60)
        logger.info("AIC-26 Ingest Pipeline — Mode B: Videos")
        logger.info(f"Input: {video_dir}")
        logger.info("=" * 60)

        # ── Step 1: Extract keyframes ─────────────────────────────────
        kf_dir = self.keyframes_dir
        logger.info(f"Step 1/5: Extracting keyframes → {kf_dir}")
        step_start = time.time()

        try:
            from aic26.pipeline.keyframe import KeyframeExtractor
            extractor = KeyframeExtractor()
            kf_stats = extractor.process(
                input_dir=video_dir,
                output_dir=kf_dir,
            )
            stats.videos_found = kf_stats.get("videos_found", 0)
            stats.keyframes_found = kf_stats.get("keyframes_total", 0)
            stats.keyframe_dir = kf_dir
            stats.steps_completed.append("keyframe_extraction")
            logger.info(
                f"[OK] Keyframe extraction done ({time.time()-step_start:.1f}s) "
                f"— {stats.videos_found} videos, {stats.keyframes_found} frames"
            )
        except Exception as e:
            logger.error(f"[FAIL] Keyframe extraction: {e}")
            stats.steps_failed.append("keyframe_extraction")
            stats.duration_seconds = time.time() - start
            return stats

        # ── Run shared processing steps ───────────────────────────────
        self._run_processing_steps(kf_dir, stats)

        stats.duration_seconds = time.time() - start
        logger.info(f"Ingestion complete: {stats}")
        return stats

    # ── Shared Steps ──────────────────────────────────────────────────

    def _run_processing_steps(self, kf_dir: str, stats: IngestStats) -> None:
        """
        Steps 2–5 shared by both input modes:
          2. Object detection
          3. OCR extraction
          4. EVA-02 embedding extraction
          5. Elasticsearch bulk indexing
        """
        # ── Step 2: Object Detection ──────────────────────────────────
        if not self.skip_detection:
            logger.info("Step 2/5: Object detection...")
            step_start = time.time()
            try:
                from aic26.pipeline.detection import ObjectDetector
                detector = ObjectDetector()
                detector.process(
                    input_dir=kf_dir,
                    output_dir=self.detection_dir,
                )
                stats.steps_completed.append("object_detection")
                logger.info(f"[OK] Detection done ({time.time()-step_start:.1f}s)")
            except Exception as e:
                logger.warning(f"[SKIP] Detection failed (non-fatal): {e}")
                stats.steps_failed.append("object_detection")
        else:
            logger.info("Step 2/5: Object detection SKIPPED")
            stats.steps_skipped.append("object_detection")

        # ── Step 3: OCR ───────────────────────────────────────────────
        if not self.skip_ocr:
            logger.info("Step 3/5: OCR extraction...")
            step_start = time.time()
            try:
                from aic26.pipeline.ocr import OCRExtractor
                ocr = OCRExtractor()
                ocr.process(
                    input_dir=kf_dir,
                    output_csv=self.ocr_path,
                )
                stats.steps_completed.append("ocr_extraction")
                logger.info(f"[OK] OCR done ({time.time()-step_start:.1f}s)")
            except Exception as e:
                logger.warning(f"[SKIP] OCR failed (non-fatal): {e}")
                stats.steps_failed.append("ocr_extraction")
        else:
            logger.info("Step 3/5: OCR extraction SKIPPED")
            stats.steps_skipped.append("ocr_extraction")

        # ── Step 4: EVA-02 Embedding ──────────────────────────────────
        logger.info("Step 4/5: EVA-02 embedding extraction...")
        step_start = time.time()
        try:
            from aic26.pipeline.embedding import EmbeddingExtractor
            embedder = EmbeddingExtractor()
            emb_stats = embedder.process_images(
                input_dir=kf_dir,
                output_dir=self.embedding_dir,
                batch_size=self.batch_size,
            )
            stats.embeddings_extracted = emb_stats.get("images_total", 0)
            stats.steps_completed.append("embedding_extraction")
            logger.info(
                f"[OK] Embeddings done ({time.time()-step_start:.1f}s) "
                f"— {stats.embeddings_extracted} images"
            )
        except Exception as e:
            logger.error(f"[FAIL] Embedding extraction: {e}")
            stats.steps_failed.append("embedding_extraction")
            return   # Cannot index without embeddings

        # ── Step 5: Elasticsearch Indexing ───────────────────────────
        if not self.skip_es_index:
            logger.info("Step 5/5: Indexing into Elasticsearch...")
            step_start = time.time()
            try:
                from aic26.core.elasticsearch_client import ElasticsearchClient
                from aic26.core.models import ModelRegistry
                from aic26.pipeline.indexer import ElasticsearchIndexer

                es = ElasticsearchClient()
                if not es.ping():
                    raise RuntimeError(
                        f"Cannot reach Elasticsearch at "
                        f"{settings.es_host}:{settings.es_port}. "
                        "Start ES with: docker compose up elasticsearch -d"
                    )

                models = ModelRegistry()
                indexer = ElasticsearchIndexer(
                    es_client=es,
                    models=models,
                    embedding_dir=self.embedding_dir,
                    keyframes_dir=kf_dir,
                    detection_dir=self.detection_dir,
                    ocr_path=self.ocr_path,
                )
                idx_stats = indexer.build_index(
                    batch_size=self.es_batch_size,
                    force_recreate=self.force_es_recreate,
                    encode_blip=self.encode_blip,
                )
                stats.es_indexed = idx_stats.indexed_success
                stats.es_failed = idx_stats.indexed_failed
                stats.steps_completed.append("es_indexing")
                logger.info(
                    f"[OK] ES indexing done ({time.time()-step_start:.1f}s) "
                    f"— {stats.es_indexed} docs indexed"
                )
            except Exception as e:
                logger.error(f"[FAIL] ES indexing: {e}")
                stats.steps_failed.append("es_indexing")
        else:
            logger.info("Step 5/5: Elasticsearch indexing SKIPPED")
            stats.steps_skipped.append("es_indexing")

    # ── Private helpers ───────────────────────────────────────────────

    def _resolve_keyframe_input(
        self,
        input_path: str,
        target_dir: str,
    ) -> str:
        """
        Resolve input to a keyframe directory:
        - If input is a .zip → extract to target_dir
        - If input is already a directory → use as-is (or copy if different)

        Returns the path to the resolved keyframe directory.
        """
        input_p = Path(input_path)

        if not input_p.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        # ZIP file → extract
        if input_p.suffix.lower() == ".zip":
            logger.info(f"Extracting ZIP: {input_path} → {target_dir}")
            os.makedirs(target_dir, exist_ok=True)
            with zipfile.ZipFile(input_path, "r") as zf:
                zf.extractall(target_dir)
            logger.info("ZIP extraction complete")

            # Find the actual keyframe folder inside extracted content
            # (might be nested, e.g. extracted/keyframes/L01_V001/)
            return self._find_keyframe_root(target_dir)

        elif input_p.is_dir():
            # Check if it's already in the right format
            if str(input_p.resolve()) == str(Path(target_dir).resolve()):
                return str(input_p)

            # Check if it's a flat keyframe folder or needs to be the root
            if self._has_video_subfolders(str(input_p)):
                logger.info(f"Using keyframe directory as-is: {input_path}")
                return str(input_p)

            # Try to find keyframe root inside
            found = self._find_keyframe_root(str(input_p))
            logger.info(f"Found keyframe root: {found}")
            return found

        else:
            raise ValueError(f"Unsupported input type: {input_path} (expected directory or .zip)")

    def _find_keyframe_root(self, base_dir: str) -> str:
        """
        Find the directory that contains video subfolders.
        Searches up to 2 levels deep.
        """
        # Level 0: base_dir itself
        if self._has_video_subfolders(base_dir):
            return base_dir

        # Level 1: immediate subdirectories
        for item in sorted(os.listdir(base_dir)):
            sub = os.path.join(base_dir, item)
            if os.path.isdir(sub) and self._has_video_subfolders(sub):
                return sub

        # Fall back to base_dir
        logger.warning(
            f"Could not find structured keyframe root under {base_dir}. "
            "Expected subfolders containing .jpg/.png files."
        )
        return base_dir

    def _has_video_subfolders(self, directory: str) -> bool:
        """Return True if directory has subfolders that contain images."""
        try:
            for item in os.listdir(directory):
                sub = os.path.join(directory, item)
                if os.path.isdir(sub) and self._list_images(sub):
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _list_video_folders(directory: str) -> List[str]:
        """List all immediate subdirectories (= video names)."""
        try:
            return sorted(
                d for d in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, d))
            )
        except Exception:
            return []

    @staticmethod
    def _list_images(directory: str) -> List[str]:
        """List all image files in a directory."""
        try:
            return [
                f for f in os.listdir(directory)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        except Exception:
            return []
