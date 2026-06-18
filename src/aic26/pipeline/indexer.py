"""
Elasticsearch indexer for AIC-26.

Reads pre-computed EVA-02 embeddings (.npy files) and paired keyframe images,
enriches each document with BLIP-ITM embeddings, object detections, OCR text,
and timing metadata, then bulk-indexes into Elasticsearch.

This replaces the legacy FAISS IndexBuilder.

Usage:
    from aic26.core.elasticsearch_client import ElasticsearchClient
    from aic26.core.models import ModelRegistry
    from aic26.pipeline.indexer import ElasticsearchIndexer

    models = ModelRegistry()
    es = ElasticsearchClient()
    es.ensure_keyframes_index()

    indexer = ElasticsearchIndexer(es, models)
    stats = indexer.build_index()
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from loguru import logger
from tqdm import tqdm

from aic26.config import settings
from aic26.core.elasticsearch_client import ElasticsearchClient
from aic26.core.models import ModelRegistry
from aic26.utils.frame_utils import parse_frame_number_from_filename
from aic26.utils.media_utils import frame_to_timestr, load_fps_from_map, load_frame_idx_from_map


@dataclass
class IndexStats:
    """Statistics from a completed index build."""
    total_videos: int = 0
    total_frames: int = 0
    indexed_success: int = 0
    indexed_failed: int = 0
    skipped_mismatch: int = 0
    blip_encoded: int = 0

    def __str__(self) -> str:
        return (
            f"IndexStats("
            f"videos={self.total_videos}, "
            f"frames={self.total_frames}, "
            f"indexed={self.indexed_success}, "
            f"failed={self.indexed_failed}, "
            f"blip_encoded={self.blip_encoded})"
        )


class ElasticsearchIndexer:
    """
    Bulk-index keyframe embeddings + metadata into Elasticsearch.

    Pipeline per video:
        1. Load EVA-02 .npy embeddings
        2. List keyframe image files
        3. For each frame:
           a. Encode with BLIP-ITM (dual embedding)
           b. Load detected_objects from detection JSON
           c. Load ocr_text from OCR CSV/JSON
           d. Load timing (fps, frame_idx_video, time_str)
        4. Bulk-index batch into ES

    Usage:
        indexer = ElasticsearchIndexer(es_client, models)
        stats = indexer.build_index(force_recreate=True)
    """

    def __init__(
        self,
        es_client: ElasticsearchClient,
        models: ModelRegistry,
        embedding_dir: Optional[str] = None,
        keyframes_dir: Optional[str] = None,
        detection_dir: Optional[str] = None,
        ocr_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            es_client: Connected ElasticsearchClient.
            models: Loaded ModelRegistry (provides BLIP encoder).
            embedding_dir: Directory with <video_name>.npy files.
            keyframes_dir: Directory with <video_name>/<frame>.jpg.
            detection_dir: (Optional) Directory with <video_name>.json detection files.
            ocr_path: (Optional) Path to OCR results CSV or JSON.
        """
        self.es = es_client
        self.models = models
        self.embedding_dir = embedding_dir or str(settings.embedding_dir)
        self.keyframes_dir = keyframes_dir or str(settings.keyframes_dir)
        self.detection_dir = detection_dir or str(settings.detection_dir)
        self.ocr_path = ocr_path or str(settings.ocr_output_path)

        # Pre-load side data
        self._detections: Dict[str, Dict[str, List[str]]] = {}
        self._ocr: Dict[str, str] = {}
        self._load_detections()
        self._load_ocr()

    # ── Public API ────────────────────────────────────────────────────────

    def build_index(
        self,
        batch_size: int = 200,
        force_recreate: bool = False,
        encode_blip: bool = True,
    ) -> IndexStats:
        """
        Index all keyframes into Elasticsearch.

        Args:
            batch_size: Documents per bulk request.
            force_recreate: Delete and recreate the ES index first.
            encode_blip: If True, encode each image with BLIP-ITM.
                         Set False to speed up indexing (single embedding).

        Returns:
            IndexStats with counts.
        """
        self.es.ensure_keyframes_index(force_recreate=force_recreate)
        stats = IndexStats()

        npy_files = sorted(
            f for f in os.listdir(self.embedding_dir) if f.endswith(".npy")
        )
        if not npy_files:
            logger.error(f"No .npy files in {self.embedding_dir}")
            return stats

        stats.total_videos = len(npy_files)
        logger.info(f"Indexing {stats.total_videos} videos into Elasticsearch...")

        batch: List[Dict[str, Any]] = []

        for npy_file in tqdm(npy_files, desc="Indexing videos"):
            video_name = os.path.splitext(npy_file)[0]
            emb_path = os.path.join(self.embedding_dir, npy_file)

            try:
                eva02_embs = np.load(emb_path).astype(np.float32)
            except Exception as e:
                logger.error(f"Failed to load {emb_path}: {e}")
                continue

            vid_folder = os.path.join(self.keyframes_dir, video_name)
            if not os.path.exists(vid_folder):
                logger.warning(f"Keyframe folder missing: {vid_folder}")
                continue

            frames = sorted(
                f for f in os.listdir(vid_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )

            n = min(len(frames), len(eva02_embs))
            if len(frames) != len(eva02_embs):
                logger.warning(
                    f"{video_name}: {len(eva02_embs)} embs vs {len(frames)} frames "
                    f"→ using min={n}"
                )
                stats.skipped_mismatch += abs(len(frames) - len(eva02_embs))

            for i in range(n):
                frame_file = frames[i]
                frame_idx = os.path.splitext(frame_file)[0]
                frame_number = parse_frame_number_from_filename(frame_idx) or i
                img_path = os.path.join(vid_folder, frame_file)
                eva02_vec = eva02_embs[i]

                # BLIP-ITM encoding (dual embedding)
                blip_vec: Optional[np.ndarray] = None
                if encode_blip and settings.use_dual_embedding:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        blip_tensor = self.models.encode_image_blip(img)
                        blip_vec = blip_tensor.cpu().numpy().astype(np.float32).flatten()
                        stats.blip_encoded += 1
                    except Exception as e:
                        logger.debug(f"BLIP encode failed for {img_path}: {e}")

                # Timing metadata
                fps = load_fps_from_map(video_name)
                frame_idx_video = (
                    load_frame_idx_from_map(video_name, frame_number)
                    if frame_number is not None else None
                )
                time_str = (
                    frame_to_timestr(frame_idx_video, fps)
                    if (fps and frame_idx_video is not None) else ""
                )

                # Object detections
                detected_objects = self._get_detections(video_name, frame_idx)

                # OCR text
                ocr_text = self._get_ocr(video_name, frame_idx)

                doc: Dict[str, Any] = {
                    "video_name": video_name,
                    "frame_idx": frame_idx,
                    "frame_number": frame_number,
                    "image_path": img_path.replace(os.sep, "/"),
                    "eva02_embedding": eva02_vec.flatten().tolist(),
                    "fps": fps,
                    "frame_idx_video": frame_idx_video,
                    "time_str": time_str,
                    "watch_url": "",
                    "detected_objects": detected_objects,
                    "ocr_text": ocr_text,
                    "description": "",   # can be filled by offline BLIP2 captioning
                }
                if blip_vec is not None:
                    doc["blip_embedding"] = blip_vec.tolist()

                batch.append(doc)
                stats.total_frames += 1

                if len(batch) >= batch_size:
                    result = self.es.bulk_index_keyframes(batch)
                    stats.indexed_success += result["success"]
                    stats.indexed_failed += result["failed"]
                    batch.clear()

        # Flush remaining batch
        if batch:
            result = self.es.bulk_index_keyframes(batch)
            stats.indexed_success += result["success"]
            stats.indexed_failed += result["failed"]

        logger.info(str(stats))
        return stats

    # ── Private helpers ───────────────────────────────────────────────────

    def _load_detections(self) -> None:
        """Load object detection JSONs from detection_dir into memory."""
        if not os.path.exists(self.detection_dir):
            logger.debug(f"Detection dir not found: {self.detection_dir}")
            return

        for fname in os.listdir(self.detection_dir):
            if not fname.endswith(".json"):
                continue
            video_name = os.path.splitext(fname)[0]
            try:
                with open(os.path.join(self.detection_dir, fname), encoding="utf-8") as f:
                    self._detections[video_name] = json.load(f)
                logger.debug(f"Loaded detections: {video_name}")
            except Exception as e:
                logger.warning(f"Failed to load detection {fname}: {e}")

    def _load_ocr(self) -> None:
        """
        Load OCR results from ocr_output_path.

        Supports CSV with columns [video_name, frame_idx, text]
        and JSON mapping 'video__frame' -> 'text'.
        """
        if not os.path.exists(self.ocr_path):
            logger.debug(f"OCR file not found: {self.ocr_path}")
            return

        try:
            if self.ocr_path.endswith(".json"):
                with open(self.ocr_path, encoding="utf-8") as f:
                    self._ocr = json.load(f)
            else:
                # Assume CSV: video_name, frame_idx, text
                import csv
                with open(self.ocr_path, encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = f"{row.get('video_name', '')}_{row.get('frame_idx', '')}"
                        self._ocr[key] = row.get("text", "")
            logger.info(f"Loaded OCR data: {len(self._ocr)} entries")
        except Exception as e:
            logger.warning(f"Failed to load OCR: {e}")

    def _get_detections(self, video_name: str, frame_idx: str) -> List[str]:
        """Return detected object names for a frame."""
        vid_dets = self._detections.get(video_name, {})
        return vid_dets.get(frame_idx, [])

    def _get_ocr(self, video_name: str, frame_idx: str) -> str:
        """Return OCR text for a frame."""
        return self._ocr.get(f"{video_name}_{frame_idx}", "")


# ── Backward-compat shim ──────────────────────────────────────────────────────

class IndexBuilder:
    """
    Backward-compatibility shim for code that imports IndexBuilder.

    New code should use ElasticsearchIndexer.
    This shim logs a deprecation warning and raises a clear error.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        logger.warning(
            "IndexBuilder (FAISS) is deprecated. "
            "Use ElasticsearchIndexer instead."
        )

    def build(self, *args: Any, **kwargs: Any) -> dict:
        raise RuntimeError(
            "IndexBuilder.build() is disabled. "
            "Use ElasticsearchIndexer.build_index() with an ElasticsearchClient."
        )