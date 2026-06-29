#!/usr/bin/env python3
"""
Smoke test for the per-keyframe metadata pipeline.

Runs `MetadataExtractor.process_all()` against the first N keyframes of a
single video folder. Designed to be runnable on a developer laptop without
a GPU: by default, the script uses **stub** detector/scene/scene-graph
callables so it can validate the orchestrator wiring end-to-end without
loading Qwen2.5-VL-7B or Places365 weights.

Pass `--real-models` to swap in the production callables (requires GPU +
pre-downloaded Places365 weights).

Usage
-----
    # Default: stub models (no GPU needed, fast)
    python scripts/smoke_test_metadata.py \\
        --video-dir data/keyframes/L01_V001 \\
        --limit 50

    # Production: load real models
    python scripts/smoke_test_metadata.py \\
        --video-dir data/keyframes/L01_V001 \\
        --limit 50 --real-models

Exit codes
----------
    0 — all keyframes processed (or skipped idempotently)
    1 — script error (bad args, missing dir, etc.)
    2 — partial failure (some keyframes failed, see logs)

The script is intentionally idempotent: re-running over an already
populated output directory skips existing sidecars unless --overwrite
is given (mirrors `MetadataExtractor.process_video` semantics).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

# Make the in-repo package importable without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ── Stub callables (default) ──────────────────────────────────────────


def _stub_detector(image_path: Path) -> dict | None:
    """Returns a tiny synthetic YOLO-style detection dict."""
    return {
        "detection_scores": [0.95],
        "detection_class_names": ["/m/01g317"],
        "detection_class_entities": ["Person"],
        "detection_boxes": [[50.0, 50.0, 200.0, 200.0]],
        "detection_class_labels": [0],
        "detection_positions": ["center"],
    }


def _stub_scene(image) -> dict | None:
    """Returns a deterministic Places365-shaped result."""
    return {
        "category": "kitchen",
        "confidence": 0.85,
        "top5": [
            ("kitchen", 0.85),
            ("dining_room", 0.07),
            ("restaurant", 0.04),
            ("living_room", 0.02),
            ("pantry", 0.01),
        ],
    }


def _stub_scene_graph(image) -> list[dict]:
    """Returns a fixed triple — proves the JSON shape without MLLM."""
    return [{"s": "person", "p": "hold", "o": "cup"}]


# ── Real callables (opt-in) ────────────────────────────────────────────


def _build_real_callables(
    settings,
) -> tuple[
    Callable[[Path], dict | None],
    Callable[[object], dict | None],
    Callable[[object], list[dict]],
]:
    """Construct detector/scene/scene-graph callables backed by the real models.

    Raises ImportError if a required dependency is missing.
    """
    import torch

    from aic26.pipeline import scene_graph as sg_mod
    from aic26.pipeline.detection import ObjectDetector
    from aic26.pipeline.places365 import Places365Classifier

    # 1. YOLO
    detector = ObjectDetector(settings)

    def _yolo(image_path: Path) -> dict | None:
        try:
            return detector._detect_single(str(image_path))
        except Exception:
            return None

    # 2. Places365
    labels_path = settings.places365_weights_path
    if labels_path is None:
        # Default path the downloader writes to (see T10).
        labels_path = Path("models/places365/categories_places365.txt")
    classifier = Places365Classifier(
        labels_path=labels_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    def _scene(image) -> dict | None:
        try:
            result = classifier.classify(image)
            return {
                "category": result.category,
                "confidence": result.confidence,
                "top5": result.top5,
            }
        except Exception:
            return None

    # 3. Qwen2.5-VL (lazy-loaded). For MVP we load it once here.
    model = _load_qwen_model(settings)

    def _sg(image) -> list[dict]:
        try:
            triples = sg_mod.extract_scene_graph(
                image,
                model=model,
                max_triples=settings.scene_graph_max_triples,
                max_new_tokens=settings.scene_graph_max_new_tokens,
            )
            return [{"s": s, "p": p, "o": o} for s, p, o in triples]
        except Exception:
            return []

    return _yolo, _scene, _sg


def _load_qwen_model(settings):
    """Load Qwen2.5-VL-7B-Instruct with 4-bit quantization (SnapSeek AD-1).

    Out of scope for MVP — returns None and lets scene_graph_fn emit []
    until R1 mitigation is in place (SmolVLM or BLIP-2 ITM fallback).
    """
    return None


# ── Main ──────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke test the per-keyframe metadata pipeline on one video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--video-dir",
        required=True,
        help="Path to one video's keyframe folder (e.g. data/keyframes/L01_V001)",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        default="data/metadata",
        help="Output root for JSON sidecars (default: data/metadata)",
    )
    p.add_argument(
        "--limit",
        "-n",
        type=int,
        default=50,
        help="Maximum keyframes to process (default: 50)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even if sidecar already exists",
    )
    p.add_argument(
        "--real-models",
        action="store_true",
        help="Use real YOLO + Places365 + Qwen (requires GPU + downloaded weights)",
    )
    p.add_argument(
        "--skip-detections",
        action="store_true",
        help="Skip YOLO pass",
    )
    p.add_argument(
        "--skip-scene",
        action="store_true",
        help="Skip Places365 pass",
    )
    p.add_argument(
        "--skip-scene-graph",
        action="store_true",
        help="Skip Qwen scene-graph pass",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    video_dir = Path(args.video_dir)
    if not video_dir.is_dir():
        print(f"❌ Not a directory: {video_dir}", file=sys.stderr)
        return 1

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Discover keyframes (cap to --limit).
    image_files = sorted(
        p
        for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )[: args.limit]

    if not image_files:
        print(f"⚠️  No keyframes in {video_dir}", file=sys.stderr)
        return 1

    print(f"📂 {video_dir}")
    print(f"   Keyframes (limit={args.limit}): {len(image_files)}")
    print(f"   Output root:                  {output_root}")
    print(f"   Mode:                         {'real-models' if args.real_models else 'stub'}")
    print()

    # Build callables.
    if args.real_models:
        from aic26.config import settings as global_settings

        try:
            det_fn, scene_fn, sg_fn = _build_real_callables(global_settings)
        except Exception as exc:
            print(f"❌ Failed to load real models: {exc}", file=sys.stderr)
            return 1
    else:
        det_fn, scene_fn, sg_fn = _stub_detector, _stub_scene, _stub_scene_graph

    # Build extractor.
    from aic26.pipeline.metadata import MetadataExtractor

    extractor = MetadataExtractor(
        detector_fn=det_fn,
        scene_fn=scene_fn,
        scene_graph_fn=sg_fn,
    )

    # Process frames one at a time so we get live progress + clear logs.
    out_dir = output_root / video_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    frames_total = len(image_files)
    frames_processed = 0
    frames_skipped = 0
    frames_failed = 0

    for i, image_path in enumerate(image_files, 1):
        sidecar_path = out_dir / f"{image_path.stem}.json"
        if sidecar_path.exists() and not args.overwrite:
            frames_skipped += 1
            continue
        try:
            meta = extractor.process_keyframe(
                image_path,
                video=video_dir.name,
                skip_detections=args.skip_detections,
                skip_scene=args.skip_scene,
                skip_scene_graph=args.skip_scene_graph,
            )
            if meta is None:
                frames_failed += 1
                continue
            meta.save(sidecar_path)
            frames_processed += 1
            elapsed = time.time() - t0
            print(
                f"  [{i:3d}/{frames_total}] {image_path.name} "
                f"→ {sidecar_path.name} ({elapsed:.1f}s)",
            )
        except KeyboardInterrupt:
            print("\n⏹  Interrupted", file=sys.stderr)
            break
        except Exception as exc:
            print(f"  [{i:3d}/{frames_total}] ❌ {image_path.name}: {exc}", file=sys.stderr)
            frames_failed += 1

    duration = time.time() - t0

    # Summary
    print()
    print("=" * 60)
    print("Smoke test summary")
    print("=" * 60)
    print(f"  Video:           {video_dir.name}")
    print(f"  Frames total:    {frames_total}")
    print(f"  Frames processed: {frames_processed}")
    print(f"  Frames skipped:  {frames_skipped} (idempotent)")
    print(f"  Frames failed:   {frames_failed}")
    print(f"  Duration:        {duration:.1f}s ({frames_processed / max(duration, 0.01):.1f} fps)")
    print(f"  Output:          {out_dir}")
    print("=" * 60)

    # Exit code: 2 if any keyframe failed; 0 otherwise.
    return 2 if frames_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
