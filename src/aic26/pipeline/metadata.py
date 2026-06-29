"""
Keyframe metadata orchestrator.

Combines per-keyframe signals from three independent passes — YOLO object
detection (with 3×3 grid position enrichment), Places365 scene
classification, and Qwen2.5-VL scene-graph extraction — into a single
JSON sidecar per keyframe.

Why a per-keyframe JSON sidecar?
--------------------------------
Mirrors the existing `data/detections/{video}/{frame}.json` layout so
keyframe metadata becomes another artifact the retriever can lazily
load on demand. Keeping each sidecar self-contained (no cross-keyframe
state) lets downstream code incrementally refresh just the affected
videos without rebuilding a global index.

Design notes
------------
- **Schema (per docs/specs/metadata-extraction.md AD-3)**:
  ```
  {
    "video":       "L01_V001",
    "frame_idx":   42,
    "detections":  [{"label", "score", "bbox", "position"}, ...],
    "scene":       {"category", "confidence", "top5": [(label, conf), ...]},
    "scene_graph": [{"s", "p", "o"}, ...]
  }
  ```
  Optional fields default to `None` (or `[]`) when their pass is skipped
  or fails — backward-compatible.
- **Injectable dependencies**: detector/scene/scene_graph callables are
  passed in by the CLI constructor (which wires up the real YOLO,
  Places365, and Qwen pipelines). Tests inject stubs returning
  deterministic dicts/tuples — no GPU, no network.
- **Idempotent re-run**: skip a sidecar that already exists (mirrors
  `PositionGrid` skip semantics from the spec's Open Question 2). Pass
  `overwrite=True` to force regeneration.
- **Graceful degradation**: a single broken keyframe does not halt the
  pipeline — log a warning, count it as failed, move on.
- **Stats**: `aggregate_metadata_stats()` walks a metadata root and
  produces scene distribution + scene-graph density — used by
  `aic26 pipeline metadata-stats`.

References
----------
- docs/specs/metadata-extraction.md §Objective + §Success Criteria.
- docs/paper-knowledge.md §5 (SnapSeek) and §2 (lifeXplore).
- docs/plans/metadata-extraction.md AD-3 (schema) + AD-4 (CLI).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


# ── JSON schema helpers ────────────────────────────────────────────────


@dataclass
class KeyframeMetadata:
    """All metadata extracted for a single keyframe.

    Attributes:
        video: Video folder name, e.g. "L01_V001".
        frame_idx: Frame index within the keyframe folder (parsed from
            filename stem, e.g. "frame_0042" → 42).
        detections: List of detection dicts:
            {"label": str, "score": float, "bbox": [x1,y1,x2,y2],
             "position": "top-left" | ... | None}.
            Empty list when detection skipped or no objects found.
        scene: Places365 top-1 result:
            {"category": str, "confidence": float,
             "top5": [(label, conf), ...]}.
            None when scene classification skipped or failed.
        scene_graph: List of 3-tuples as {s, p, o} dicts.
            None when scene-graph extraction skipped or failed.
    """

    video: str
    frame_idx: int
    detections: list[dict[str, Any]]
    scene: dict[str, Any] | None
    scene_graph: list[dict[str, Any]] | None

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-friendly dict.

        `detections` is always an array (possibly empty) since detection
        is the cheap pass and consumers expect a list. `scene` and
        `scene_graph` preserve None when not run / unreadable, so
        downstream code can distinguish "skipped" (null) from "ran but
        produced nothing" (empty array).
        """
        return {
            "video": self.video,
            "frame_idx": self.frame_idx,
            "detections": list(self.detections or []),
            "scene": self.scene,
            "scene_graph": (list(self.scene_graph) if self.scene_graph is not None else None),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> KeyframeMetadata:
        """Build a KeyframeMetadata from a JSON-decoded dict."""
        return cls(
            video=raw["video"],
            frame_idx=int(raw["frame_idx"]),
            detections=list(raw.get("detections") or []),
            scene=raw.get("scene"),
            scene_graph=list(raw.get("scene_graph") or []),
        )

    @classmethod
    def from_json(cls, raw: str) -> KeyframeMetadata:
        """Build a KeyframeMetadata from a JSON string."""
        return cls.from_dict(json.loads(raw))

    def save(self, path: Path, indent: int = 2) -> None:
        """Write this metadata to a JSON file at `path` (creating parents)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=indent), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> KeyframeMetadata:
        """Read metadata from a JSON file."""
        return cls.from_json(path.read_text(encoding="utf-8"))


# ── Stats aggregate ───────────────────────────────────────────────────


@dataclass
class MetadataStats:
    """Aggregate view of a metadata root directory.

    Attributes:
        total_sidecars: Number of sidecar JSON files processed.
        total_videos:   Number of video folders (sub-directories) found.
        scene_distribution: Counter of scene-category → frame count.
            Only counts frames with `scene` populated and a non-empty
            `category` string.
        scene_graph_total_triples: Total sum of triple counts across
            all sidecars that have scene_graph populated.
        avg_triples_per_keyframe: scene_graph_total_triples /
            number of keyframes that have a scene_graph (0.0 if none).
    """

    total_sidecars: int = 0
    total_videos: int = 0
    scene_distribution: dict[str, int] = field(default_factory=dict)
    scene_graph_total_triples: int = 0
    avg_triples_per_keyframe: float = 0.0

    def __repr__(self) -> str:
        return (
            f"MetadataStats(sidecars={self.total_sidecars}, "
            f"videos={self.total_videos}, "
            f"top_scene={self._top_scene_str()})"
        )

    def _top_scene_str(self) -> str:
        if not self.scene_distribution:
            return "n/a"
        top, n = next(iter(self.scene_distribution.items()))
        return f"{top}({n})"


def aggregate_metadata_stats(metadata_root: Path) -> MetadataStats:
    """Walk a metadata root directory and aggregate per-frame stats.

    Args:
        metadata_root: Path to `data/metadata/` (which itself contains
            one subdirectory per video, each holding frame JSON files).

    Returns:
        MetadataStats populated with counts + scene distribution +
        scene-graph triple total.
    """
    metadata_root = Path(metadata_root)
    stats = MetadataStats()

    if not metadata_root.exists():
        return stats

    video_dirs = sorted(p for p in metadata_root.iterdir() if p.is_dir())
    stats.total_videos = len(video_dirs)

    scene_counts: Counter[str] = Counter()
    total_triples = 0
    keyframes_with_scene_graph = 0
    sidecar_count = 0

    for vdir in video_dirs:
        for sidecar_path in sorted(vdir.glob("*.json")):
            sidecar_count += 1
            try:
                m = KeyframeMetadata.load(sidecar_path)
            except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
                logger.warning("Skipping malformed sidecar %s: %s", sidecar_path, exc)
                continue
            if m.scene and isinstance(m.scene.get("category"), str):
                scene_counts[m.scene["category"]] += 1
            if m.scene_graph:
                total_triples += len(m.scene_graph)
                keyframes_with_scene_graph += 1

    stats.total_sidecars = sidecar_count
    stats.scene_distribution = dict(scene_counts.most_common())
    stats.scene_graph_total_triples = total_triples
    if keyframes_with_scene_graph > 0:
        stats.avg_triples_per_keyframe = total_triples / keyframes_with_scene_graph
    return stats


# ── Helpers ────────────────────────────────────────────────────────────


_IMAGE_SUFFIXES: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".webp",
)


def _frame_idx_from_path(image_path: Path) -> int:
    """Extract the trailing integer from a keyframe filename.

    Examples:
        "frame_0042.jpg"           → 42
        "keyframe_1234.png"        → 1234
        "L01_V001_0007.jpg"        → 7  (right-most integer)

    Returns 0 if no integer can be parsed.
    """
    import re

    stem = image_path.stem
    matches = re.findall(r"\d+", stem)
    return int(matches[-1]) if matches else 0


def _safe_load_image(image_path: Path) -> Image.Image | None:
    """Open a keyframe as RGB; return None on corrupt/unreadable input."""
    try:
        img = Image.open(image_path)
        img.load()
        return img.convert("RGB") if img.mode != "RGB" else img
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Cannot open image %s: %s", image_path, exc)
        return None


def _normalize_detector_output(det: dict | None) -> list[dict]:
    """Convert raw YOLO-output dict → flat list of detection records.

    Each output record has: label, score, bbox, position.
    """
    if not det:
        return []
    names = det.get("detection_class_names") or []
    entities = det.get("detection_class_entities") or []
    scores = det.get("detection_scores") or []
    boxes = det.get("detection_boxes") or []
    positions = det.get("detection_positions") or []
    n = len(boxes)
    # Cross-check lengths — downstream consumers expect parallel arrays.
    if not (len(scores) == len(boxes) == len(entities) == len(names) == n):
        logger.debug(
            "Detection dict has inconsistent lengths (%d); skipping normalization",
            n,
        )
        return []
    out = []
    for i in range(n):
        # Prefer human-readable entity over MID (e.g. "Person" over "/m/01g317").
        label = entities[i] if entities[i] else names[i]
        out.append(
            {
                "label": str(label),
                "score": float(scores[i]),
                "bbox": [float(x) for x in boxes[i]],
                "position": positions[i] if i < len(positions) else None,
            }
        )
    return out


# ── Orchestrator ───────────────────────────────────────────────────────


# Type aliases for injected callables. These let the CLI swap real
# models in while tests use deterministic stubs.
DetectorFn = Callable[[Path], dict | None]
SceneFn = Callable[[Image.Image], dict | None]
SceneGraphFn = Callable[[Image.Image], list[dict]]


class MetadataExtractor:
    """Orchestrator that combines detection + scene + scene-graph into JSON sidecars.

    Args:
        detector_fn: Callable taking an image Path and returning a YOLO
            detection dict with parallel arrays (see pipeline/detection.py).
            Returns dict on success, None on failure.
        scene_fn: Callable taking a PIL Image and returning a Places365
            result dict ({"category", "confidence", "top5"}) or None.
        scene_graph_fn: Callable taking a PIL Image and returning a list
            of {s, p, o} dicts (often produced by
            aic26.pipeline.scene_graph.extract_scene_graph).
    """

    def __init__(
        self,
        detector_fn: DetectorFn,
        scene_fn: SceneFn,
        scene_graph_fn: SceneGraphFn,
    ) -> None:
        self.detector_fn = detector_fn
        self.scene_fn = scene_fn
        self.scene_graph_fn = scene_graph_fn

    # ── Single keyframe ──────────────────────────────────────────

    def process_keyframe(
        self,
        image_path: Path,
        video: str | None = None,
        frame_idx: int | None = None,
        skip_detections: bool = False,
        skip_scene: bool = False,
        skip_scene_graph: bool = False,
    ) -> KeyframeMetadata | None:
        """Run all three passes on one keyframe; return assembled metadata.

        Returns None only if the image cannot be loaded at all.
        """
        image_path = Path(image_path)
        if video is None:
            # Best-effort derive from parent folder name.
            video = image_path.parent.name
        if frame_idx is None:
            frame_idx = _frame_idx_from_path(image_path)

        img = _safe_load_image(image_path)
        if img is None:
            # Corrupt / unreadable image: caller (process_video) will
            # count this as failed. Don't write a sidecar — there's no
            # trustworthy content to record.
            return None

        detections: list[dict[str, Any]] = []
        if not skip_detections:
            try:
                detections = _normalize_detector_output(self.detector_fn(image_path))
            except Exception as exc:  # noqa: BLE001 — log+continue
                logger.warning("Detection failed for %s: %s", image_path, exc)
                detections = []

        scene: dict[str, Any] | None = None
        if not skip_scene:
            try:
                scene = self.scene_fn(img)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Scene classification failed for %s: %s", image_path, exc)
                scene = None

        scene_graph: list[dict[str, Any]] | None
        if skip_scene_graph:
            scene_graph = None
        else:
            try:
                triples = self.scene_graph_fn(img) or []
                # Model ran — preserve empty list to distinguish from
                # skipped (None) downstream.
                scene_graph = list(triples)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Scene graph extraction failed for %s: %s", image_path, exc)
                scene_graph = None

        return KeyframeMetadata(
            video=video,
            frame_idx=frame_idx,
            detections=detections,
            scene=scene,
            scene_graph=scene_graph,
        )

    # ── One video folder ──────────────────────────────────────────

    def process_video(
        self,
        video_dir: Path,
        output_root: Path,
        video_name: str | None = None,
        skip_detections: bool = False,
        skip_scene: bool = False,
        skip_scene_graph: bool = False,
        overwrite: bool = False,
    ) -> dict[str, int]:
        """Walk one video's keyframe folder and emit sidecars.

        Args:
            video_dir: Path to the video's keyframe folder (containing
                ``*.jpg``/``*.png`` files).
            output_root: Root for output; sidecars will be written to
                ``output_root / video_name / <stem>.json``.
            video_name: Override video folder name (else derived from
                ``video_dir.name``).
            skip_*: Pass-through flags to ``process_keyframe``.
            overwrite: If False (default), skip sidecars that already
                exist on disk.

        Returns:
            Stats dict with keys: ``frames_total``, ``frames_processed``,
            ``frames_skipped``, ``frames_failed``.
        """
        video_dir = Path(video_dir)
        output_root = Path(output_root)
        if video_name is None:
            video_name = video_dir.name

        image_files = sorted(
            p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )

        frames_total = len(image_files)
        frames_processed = 0
        frames_skipped = 0
        frames_failed = 0

        out_dir = output_root / video_name

        for image_path in image_files:
            sidecar_path = out_dir / f"{image_path.stem}.json"
            if sidecar_path.exists() and not overwrite:
                frames_skipped += 1
                continue

            meta = self.process_keyframe(
                image_path,
                video=video_name,
                skip_detections=skip_detections,
                skip_scene=skip_scene,
                skip_scene_graph=skip_scene_graph,
            )
            if meta is None:
                frames_failed += 1
                continue
            try:
                meta.save(sidecar_path)
                frames_processed += 1
            except OSError as exc:
                logger.warning("Failed to write %s: %s", sidecar_path, exc)
                frames_failed += 1

        return {
            "frames_total": frames_total,
            "frames_processed": frames_processed,
            "frames_skipped": frames_skipped,
            "frames_failed": frames_failed,
        }

    # ── All video folders ─────────────────────────────────────────

    def process_all(
        self,
        keyframes_root: Path,
        output_root: Path,
        skip_detections: bool = False,
        skip_scene: bool = False,
        skip_scene_graph: bool = False,
        overwrite: bool = False,
    ) -> dict[str, int]:
        """Walk every video folder under ``keyframes_root`` and process each.

        Returns:
            Stats dict with: ``videos_total``, ``videos_processed``,
            ``frames_total``, ``frames_processed``, ``frames_skipped``,
            ``frames_failed``.
        """
        keyframes_root = Path(keyframes_root)
        output_root = Path(output_root)

        video_dirs = sorted(p for p in keyframes_root.iterdir() if p.is_dir())
        videos_total = len(video_dirs)

        agg_frames_total = 0
        agg_frames_processed = 0
        agg_frames_skipped = 0
        agg_frames_failed = 0
        videos_processed = 0

        for video_dir in video_dirs:
            stats = self.process_video(
                video_dir=video_dir,
                output_root=output_root,
                skip_detections=skip_detections,
                skip_scene=skip_scene,
                skip_scene_graph=skip_scene_graph,
                overwrite=overwrite,
            )
            agg_frames_total += stats["frames_total"]
            agg_frames_processed += stats["frames_processed"]
            agg_frames_skipped += stats["frames_skipped"]
            agg_frames_failed += stats["frames_failed"]
            if stats["frames_total"] > 0:
                videos_processed += 1

        return {
            "videos_total": videos_total,
            "videos_processed": videos_processed,
            "frames_total": agg_frames_total,
            "frames_processed": agg_frames_processed,
            "frames_skipped": agg_frames_skipped,
            "frames_failed": agg_frames_failed,
        }


__all__ = [
    "KeyframeMetadata",
    "MetadataStats",
    "MetadataExtractor",
    "aggregate_metadata_stats",
]
