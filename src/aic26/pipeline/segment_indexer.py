"""
Segment Index Builder for AIC-26 (Stage-1 of hierarchical two-stage search).

Reads video description JSON files from `data/vid-description/`, formats
structured text (description + category + keywords + entities), embeds them
with BAAI/bge-m3, and builds a FAISS IndexFlatIP with full metadata.

Output: data/faiss/segment_index.pkl

JSON format supported (both variants):
    Variant A (spec):   {"video_id": "L29_V001", "segments": [...]}
    Variant B (actual): {"L30_V001": {"segments": [...]}, ...}

Device policy:
    Offline indexing ALWAYS prefers GPU (cuda) for maximum throughput.
    This is independent of the ``TEXT_RETRIEVAL`` setting, which only
    controls the online retrieval server.

References:
    - Spec: docs/specs/two-stage-hierarchical-search.md §2, §4.2
"""
from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm

from aic26.config import settings
from aic26.config.constants import BGE_M3_EMBEDDING_DIM


def format_segment_text(segment: dict) -> str:
    """
    Format a segment dict into a structured text string for BGE-M3 embedding.

    Combines description, category, keywords, and entities using pipe separation.

    Args:
        segment: Segment dict with keys: description, category, keywords, entities.

    Returns:
        Formatted string like:
        "desc text | Thể loại: category | Từ khóa: kw1, kw2 | Thực thể: e1, e2"
    """
    desc = segment.get("description", "").strip()
    category = segment.get("category", "").strip()
    keywords = segment.get("keywords", [])
    entities = segment.get("entities", [])

    parts = [desc] if desc else []
    if category:
        parts.append(f"Thể loại: {category}")
    if keywords:
        kw_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
        parts.append(f"Từ khóa: {kw_str}")
    if entities:
        ent_str = ", ".join(entities) if isinstance(entities, list) else str(entities)
        parts.append(f"Thực thể: {ent_str}")

    return " | ".join(filter(None, parts))


class SegmentIndexBuilder:
    """
    Reads vid-description JSON files, embeds segments via BGE-M3, and builds
    a FAISS IndexFlatIP + metadata dict saved as `segment_index.pkl`.

    Usage:
        builder = SegmentIndexBuilder()
        stats = builder.build(
            input_dir="data/vid-description",
            output_path="data/faiss/segment_index.pkl",
            batch_size=64,
        )
    """

    def __init__(
        self,
        model_name: str = settings.bge_m3_model_name,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name: HuggingFace model id for BGE-M3.
            device: Override device string ('cpu', 'cuda', 'cuda:0').
                    Defaults to 'cuda' if CUDA is available, otherwise 'cpu'.
                    Offline indexing should always run on GPU for speed;
                    this default enforces that without depending on
                    the TEXT_RETRIEVAL env var (which is for online server).
        """
        self.model_name = model_name
        if device is not None:
            self.device = device
        else:
            import torch  # lazy import: avoid loading libtorch_cuda at CLI startup
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"[SegmentIndexBuilder] Will embed with BGE-M3 on device='{self.device}' "
            f"(offline indexing always prefers GPU)."
        )


    # ── Public ────────────────────────────────────────────────────────

    def build(
        self,
        input_dir: Optional[str] = None,
        output_path: Optional[str] = None,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """
        Build and persist the segment FAISS index.

        Args:
            input_dir: Directory containing vid-description JSON files.
            output_path: Path for the output segment_index.pkl.
            batch_size: Number of segments per BGE-M3 forward pass.

        Returns:
            Stats dict with keys: total_videos, total_segments, output_path.
        """
        import faiss

        from aic26.core.text_embedding import BGEM3TextEmbedder

        input_dir = input_dir or str(settings.vid_description_dir)
        output_path = output_path or str(settings.segment_index_path)

        # Ensure output dir exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 1. Load all segments from JSON files
        logger.info(f"Loading video descriptions from '{input_dir}'...")
        all_segments, all_texts = self._load_all_segments(input_dir)

        if not all_segments:
            raise RuntimeError(
                f"No segments found in '{input_dir}'. "
                "Ensure vid-description JSON files exist."
            )

        logger.info(
            f"Loaded {len(all_segments)} segments from "
            f"{len(set(s['video_id'] for s in all_segments))} videos."
        )

        # 2. Embed all segment texts
        logger.info(f"Embedding {len(all_texts)} segments with BGE-M3...")
        embedder = BGEM3TextEmbedder(model_name=self.model_name, device=self.device)
        embeddings = embedder.embed_texts(all_texts, batch_size=batch_size)  # (N, 1024)

        # 3. Build FAISS IndexFlatIP (cosine sim with normalized vectors = dot product)
        logger.info("Building FAISS IndexFlatIP(1024)...")
        index = faiss.IndexFlatIP(BGE_M3_EMBEDDING_DIM)
        index.add(embeddings)

        # 4. Build video_to_segment_indices mapping
        video_to_segment_indices: Dict[str, List[int]] = defaultdict(list)
        for i, seg in enumerate(all_segments):
            video_to_segment_indices[seg["video_id"]].append(i)

        # 5. Pack and persist
        index_data = {
            "faiss_index": index,
            "embeddings": embeddings,
            "segments": all_segments,
            "video_to_segment_indices": dict(video_to_segment_indices),
            "model_name": self.model_name,
            "embedding_dim": BGE_M3_EMBEDDING_DIM,
        }

        with open(output_path, "wb") as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        stats = {
            "total_videos": len(video_to_segment_indices),
            "total_segments": len(all_segments),
            "embedding_shape": embeddings.shape,
            "output_path": output_path,
            "file_size_mb": os.path.getsize(output_path) / 1_048_576,
        }
        logger.info(
            f"Segment index built: {stats['total_segments']} segments, "
            f"{stats['total_videos']} videos → {output_path} "
            f"({stats['file_size_mb']:.1f} MB)"
        )
        return stats

    # ── Private ───────────────────────────────────────────────────────

    def _load_all_segments(
        self, input_dir: str
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Parse all JSON files in input_dir and collect segment metadata + texts.

        Handles two JSON formats:
          - Variant A (spec):   {"video_id": "...", "segments": [...]}
          - Variant B (actual): {"L30_V001": {"segments": [...]}, "L30_V002": {...}}

        Returns:
            (all_segments, all_texts) where all_segments is a list of metadata
            dicts and all_texts is the corresponding list of formatted strings.
        """
        all_segments: List[Dict[str, Any]] = []
        all_texts: List[str] = []

        json_files = sorted(Path(input_dir).glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files found in '{input_dir}'")
            return all_segments, all_texts

        for json_path in tqdm(json_files, desc="Loading JSON descriptions", unit="file"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load {json_path}: {e}")
                continue

            file_segments, file_texts = self._parse_json(data, json_path.name)
            all_segments.extend(file_segments)
            all_texts.extend(file_texts)

        return all_segments, all_texts

    def _parse_json(
        self, data: Any, filename: str
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Parse a single JSON object into segment metadata + formatted texts.

        Supports:
          - Variant A: {"video_id": "L29_V001", "segments": [...]}
          - Variant B: {"L30_V001": {"segments": [...]}, ...}
        """
        segments: List[Dict[str, Any]] = []
        texts: List[str] = []

        if not isinstance(data, dict):
            logger.warning(f"Unexpected JSON format in {filename}: expected dict")
            return segments, texts

        # ── Variant A: top-level "video_id" key ───────────────────────
        if "video_id" in data and "segments" in data:
            video_id = data["video_id"]
            seg_list = data.get("segments", [])
            for i, seg in enumerate(seg_list):
                meta, text = self._build_segment_meta(video_id, i, seg)
                segments.append(meta)
                texts.append(text)
            return segments, texts

        # ── Variant B: dict of video_id -> {segments: [...]} ──────────
        for video_id, video_data in data.items():
            if not isinstance(video_data, dict):
                continue
            seg_list = video_data.get("segments", [])
            if not isinstance(seg_list, list):
                continue
            for i, seg in enumerate(seg_list):
                meta, text = self._build_segment_meta(video_id, i, seg)
                segments.append(meta)
                texts.append(text)

        return segments, texts

    def _build_segment_meta(
        self, video_id: str, local_idx: int, seg: dict
    ) -> Tuple[Dict[str, Any], str]:
        """Build a normalized segment metadata dict and its formatted text."""
        formatted = format_segment_text(seg)
        meta = {
            "video_id": video_id,
            "start_time": seg.get("start_time", ""),
            "end_time": seg.get("end_time", ""),
            "description": seg.get("description", ""),
            "category": seg.get("category", ""),
            "entities": seg.get("entities", []),
            "keywords": seg.get("keywords", []),
            "formatted_text": formatted,
        }
        return meta, formatted
