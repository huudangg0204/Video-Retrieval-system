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
             "top5": [(label, conf), ...],
             "scene_group": str (optional high-level group)}.
            None when scene classification skipped or failed.
        scene_graph: List of 3-tuples as {s, p, o} dicts.
            None when scene-graph extraction skipped or failed.
        scene_group: Coarse scene group (e.g. "outdoor_urban",
            "indoor_residential"). Derived from scene category.
            None when scene is None.
        smoothed_scene: Scene category after temporal smoothing.
            Populated by ``apply_temporal_smoothing()`` post-processing.
            None when smoothing has not been applied.
    """

    video: str
    frame_idx: int
    detections: list[dict[str, Any]]
    scene: dict[str, Any] | None
    scene_graph: list[dict[str, Any]] | None
    scene_group: str | None = None
    smoothed_scene: str | None = None

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-friendly dict.

        `detections` is always an array (possibly empty) since detection
        is the cheap pass and consumers expect a list. `scene` and
        `scene_graph` preserve None when not run / unreadable, so
        downstream code can distinguish "skipped" (null) from "ran but
        produced nothing" (empty array).
        """
        d: dict[str, Any] = {
            "video": self.video,
            "frame_idx": self.frame_idx,
            "detections": list(self.detections or []),
            "scene": self.scene,
            "scene_graph": (list(self.scene_graph) if self.scene_graph is not None else None),
        }
        # Only serialize Phase 2 fields when populated (backward-compatible).
        if self.scene_group is not None:
            d["scene_group"] = self.scene_group
        if self.smoothed_scene is not None:
            d["smoothed_scene"] = self.smoothed_scene
        return d

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
            scene_group=raw.get("scene_group"),
            smoothed_scene=raw.get("smoothed_scene"),
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


# ── Temporal smoothing ─────────────────────────────────────────────────


def apply_temporal_smoothing(
    metadata_dir: Path,
    video_name: str | None = None,
    window_size: int = 5,
    overwrite_files: bool = True,
) -> dict[str, int]:
    """Apply sliding-window majority vote to smooth scene labels.

    Adjacent keyframes within a video almost always depict the same scene.
    Per-frame Places365 classifications can be noisy — e.g. one frame in a
    continuous street segment might be classified as "highway" while its
    neighbors say "street". Temporal smoothing replaces each frame's scene
    category with the majority category within a centered window of
    ``window_size`` frames.

    This is a **post-processing** step that operates on already-written
    sidecar JSONs. It reads all sidecars for a video, computes smoothed
    labels, writes the ``smoothed_scene`` field back, and returns stats.

    Args:
        metadata_dir: Root metadata directory (contains video sub-folders).
        video_name: If set, only smooth this one video. If None, smooth
            all video sub-folders.
        window_size: Number of frames in the sliding window. Must be odd
            (if even, +1 is applied). Default 5.
        overwrite_files: If True, write the updated sidecars back to disk.

    Returns:
        Stats dict: ``smoothed_videos``, ``smoothed_frames``,
        ``categories_before``, ``categories_after``.
    """
    metadata_dir = Path(metadata_dir)
    if window_size < 1:
        window_size = 1
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd for symmetric window.
    half = window_size // 2

    video_dirs: list[Path]
    if video_name:
        video_dirs = [metadata_dir / video_name]
    else:
        video_dirs = sorted(p for p in metadata_dir.iterdir() if p.is_dir())

    total_smoothed_frames = 0
    total_smoothed_videos = 0
    categories_before: set[str] = set()
    categories_after: set[str] = set()

    for vdir in video_dirs:
        sidecar_paths = sorted(vdir.glob("*.json"))
        if not sidecar_paths:
            continue

        # Load all sidecars for this video.
        sidecars: list[tuple[Path, KeyframeMetadata]] = []
        for sp in sidecar_paths:
            try:
                m = KeyframeMetadata.load(sp)
                sidecars.append((sp, m))
            except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
                logger.warning("Skip malformed sidecar %s: %s", sp, exc)
                continue

        if not sidecars:
            continue

        # Extract scene labels (raw categories).
        raw_scenes = [
            m.scene.get("category", "") if m.scene else ""
            for _, m in sidecars
        ]

        # Collect pre-smoothing category set.
        categories_before.update(c for c in raw_scenes if c)

        # Sliding window majority vote.
        n = len(raw_scenes)
        smoothed = []
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            window_labels = [l for l in raw_scenes[start:end] if l]
            if window_labels:
                vote_counter: Counter[str] = Counter(window_labels)
                majority = vote_counter.most_common(1)[0][0]
                smoothed.append(majority)
            else:
                smoothed.append(raw_scenes[i] if raw_scenes[i] else None)

        # Write back.
        for (sp, m), sm_label in zip(sidecars, smoothed):
            if sm_label:
                m.smoothed_scene = sm_label
                categories_after.add(sm_label)
                if overwrite_files:
                    m.save(sp)
                total_smoothed_frames += 1

        total_smoothed_videos += 1

    return {
        "smoothed_videos": total_smoothed_videos,
        "smoothed_frames": total_smoothed_frames,
        "categories_before": len(categories_before),
        "categories_after": len(categories_after),
    }

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


# ── Factory functions for real model callables ─────────────────────────


def _enrich_positions_from_image(
    det: dict,
    image_path: Path,
    min_bbox_fraction: float,
) -> dict:
    """Compute ``detection_positions`` for a detection dict missing the field.

    Older detection JSONs (produced before position enrichment was added to
    ``ObjectDetector``) lack the ``detection_positions`` key. This helper
    reads the keyframe's dimensions from ``image_path`` and computes the
    3×3 grid label for every bbox via
    ``pipeline.detection.add_position_to_detections`` — the same helper
    ``ObjectDetector._detect_single`` uses today.

    Idempotency: if the input dict already carries ``detection_positions``,
    it is returned unchanged. The detection writer should never overwrite
    existing positions.

    Defensive: corrupt or missing image → return ``det`` unchanged. The
    caller will still record detections (with ``position: None``) rather
    than dropping them entirely.

    Args:
        det: Detection dict (parallel arrays — see ``_normalize_detector_output``).
        image_path: Path to the matching keyframe image; only used to read
            width/height.
        min_bbox_fraction: Forwarded to ``position_from_bbox`` (default 0.05
            from ``POSITION_GRID_THRESHOLD``).

    Returns:
        Detection dict, possibly with a new ``detection_positions`` field.
    """
    if "detection_positions" in det:
        return det
    if not det.get("detection_boxes"):
        return det
    try:
        from aic26.pipeline.detection import add_position_to_detections

        with Image.open(image_path) as img:
            img_w, img_h = img.size
        return add_position_to_detections(
            det,
            img_h=img_h,
            img_w=img_w,
            min_bbox_fraction=min_bbox_fraction,
        )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        # Defensive: corrupt image → return as-is. Caller will record
        # position=None for every detection, which matches the previous
        # behavior and is better than dropping detections entirely.
        logger.debug("Position enrichment skipped for %s: %s", image_path, exc)
        return det


def load_precomputed_detections(
    detection_dir: Path,
    min_bbox_fraction: float | None = None,
) -> "DetectorFn":
    """Create a DetectorFn that reads pre-computed YOLO detection JSONs.

    Instead of running YOLO inference again, this reads existing detection
    results from ``data/detections/{video}/{frame}.json`` files produced
    by the pipeline's Step 2 (``ObjectDetector``).

    For detection files produced by an older ``ObjectDetector`` version
    that did not compute the 3×3 grid position, this factory also lazily
    enriches the dict on first read by opening the matching keyframe
    (``image_path``) and calling ``add_position_to_detections``. Existing
    positions are never overwritten.

    Args:
        detection_dir: Root directory containing per-video detection
            sub-folders (e.g. ``data/detections/``).
        min_bbox_fraction: Minimum bbox area as fraction of image area to
            emit a position label (else None). Defaults to
            ``POSITION_GRID_THRESHOLD`` (0.05). Forwarded to
            ``add_position_to_detections`` only when the source dict lacks
            ``detection_positions``.

    Returns:
        A callable ``(image_path: Path) -> dict | None`` suitable for
        ``MetadataExtractor(detector_fn=...)``.
    """
    from aic26.config.constants import POSITION_GRID_THRESHOLD

    detection_dir = Path(detection_dir)
    if min_bbox_fraction is None:
        min_bbox_fraction = POSITION_GRID_THRESHOLD

    def _detector_fn(image_path: Path) -> dict | None:
        stem = image_path.stem
        video = image_path.parent.name
        det_path = detection_dir / video / f"{stem}.json"
        if det_path.exists():
            try:
                det = json.loads(det_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Failed to read pre-computed detection %s: %s", det_path, exc
                )
                return None
            # Backfill positions for older detection JSONs that lack the field.
            return _enrich_positions_from_image(
                det, image_path, min_bbox_fraction=min_bbox_fraction
            )
        logger.debug("No pre-computed detection for %s/%s", video, stem)
        return None

    return _detector_fn


def build_metadata_callables(
    detection_dir: Path,
    places365_weights_path: Path | None = None,
    places365_labels_path: Path | None = None,
    scene_graph_model_name: str | None = None,
    device: str = "cpu",
    scene_graph_max_triples: int = 5,
    scene_graph_max_new_tokens: int = 512,
) -> tuple["DetectorFn", "SceneFn", "SceneGraphFn"]:
    """Build real detector/scene/scene-graph callables for production use.

    This is the canonical factory used by both ``orchestrator.py`` and
    ``cli/main.py::extract_metadata_cmd`` to wire up real models instead
    of dummy lambdas.

    Args:
        detection_dir: Root of pre-computed detection JSONs (e.g.
            ``data/detections/``).
        places365_weights_path: Path to ``resnet50_places365.pth.tar``.
            If None or doesn't exist, scene_fn degrades gracefully to
            return None.
        places365_labels_path: Path to ``categories_places365.txt``.
            If None, auto-resolved relative to weights_path.
        scene_graph_model_name: Qwen model id for scene-graph extraction.
            If None, scene_graph_fn degrades gracefully to return [].
        device: Torch device string ('cuda', 'cpu', ...).
        scene_graph_max_triples: Max triples per image.
        scene_graph_max_new_tokens: Max tokens for scene-graph model.

    Returns:
        Tuple of (detector_fn, scene_fn, scene_graph_fn).
    """
    # 1. Detection: always read from pre-computed files
    detector_fn = load_precomputed_detections(detection_dir)

    # 2. Scene classification: Places365
    scene_fn: SceneFn = lambda img: None  # noqa: E731 — default fallback
    if places365_weights_path is not None:
        weights_p = Path(places365_weights_path)
        if weights_p.is_dir():
            weights_p = weights_p / "resnet50_places365.pth.tar"
        if weights_p.exists():
            labels_p = (
                Path(places365_labels_path)
                if places365_labels_path
                else weights_p.parent / "categories_places365.txt"
            )
            if labels_p.exists():
                try:
                    from aic26.pipeline.places365 import Places365Classifier

                    classifier = Places365Classifier.from_weights(
                        weights_path=weights_p,
                        labels_path=labels_p,
                        device=device,
                    )
                    logger.info(
                        "Places365 classifier loaded from %s (device=%s)",
                        weights_p,
                        device,
                    )

                    def _scene_fn(image: Image.Image) -> dict | None:
                        try:
                            from aic26.pipeline.places365 import _infer_scene_group

                            result = classifier.classify(image)
                            category = result.category
                            confidence = result.confidence

                            # Confidence thresholding: low-confidence scenes
                            # are labeled "uncertain" to reduce noise.
                            if confidence < 0.3:
                                category = "uncertain"
                                # Infer scene_group from top-5 via majority
                                # vote — even if exact category is uncertain,
                                # the group (indoor/outdoor/...) is useful.
                                top5_groups = [
                                    _infer_scene_group(lbl) for lbl, _ in result.top5
                                ]
                                group_counter = Counter(top5_groups)
                                scene_group = group_counter.most_common(1)[0][0]
                            else:
                                scene_group = _infer_scene_group(category)

                            return {
                                "category": category,
                                "confidence": confidence,
                                "top5": result.top5,
                                "scene_group": scene_group,
                            }
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Places365 classify failed: %s", exc)
                            return None

                    scene_fn = _scene_fn
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to load Places365 classifier: %s — scene disabled",
                        exc,
                    )
            else:
                logger.warning("Places365 labels not found at %s", labels_p)
        else:
            logger.warning("Places365 weights not found at %s", weights_p)

    # 3. Scene graph: Qwen2.5-VL (heavy, optional)
    scene_graph_fn: SceneGraphFn = lambda img: []  # noqa: E731 — default fallback
    if scene_graph_model_name:
        try:
            from aic26.pipeline.scene_graph import extract_scene_graph

            # Lazy-load the model on first call to avoid VRAM allocation
            # when the scene-graph step is skipped via --skip-scene-graph.
            _qwen_model = None
            _model_name = scene_graph_model_name

            def _sg_fn(image: Image.Image) -> list[dict]:
                nonlocal _qwen_model
                if _qwen_model is None:
                    logger.info(
                        "Lazy-loading Qwen scene-graph model: %s", _model_name
                    )
                    try:
                        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                        _qwen_model_obj = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                            _model_name,
                            device_map="auto",
                            torch_dtype="auto",
                        )
                        _qwen_model = _qwen_model_obj
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Failed to load Qwen model %s: %s — scene graph disabled",
                            _model_name,
                            exc,
                        )
                        return []
                try:
                    triples = extract_scene_graph(
                        image,
                        model=_qwen_model,
                        max_triples=scene_graph_max_triples,
                        max_new_tokens=scene_graph_max_new_tokens,
                    )
                    return [{"s": s, "p": p, "o": o} for s, p, o in triples]
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Scene graph extraction failed: %s", exc)
                    return []

            scene_graph_fn = _sg_fn
        except ImportError:
            logger.warning(
                "scene_graph module not available — scene graph disabled"
            )

    return detector_fn, scene_fn, scene_graph_fn


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

        # Derive scene_group from scene result (if populated by enriched scene_fn).
        scene_group: str | None = None
        if scene is not None:
            scene_group = scene.get("scene_group")

        return KeyframeMetadata(
            video=video,
            frame_idx=frame_idx,
            detections=detections,
            scene=scene,
            scene_graph=scene_graph,
            scene_group=scene_group,
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
    "apply_temporal_smoothing",
    "load_precomputed_detections",
    "build_metadata_callables",
]
