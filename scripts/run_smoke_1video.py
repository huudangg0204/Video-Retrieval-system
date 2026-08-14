"""
Minimal pipeline smoke test — 1 video, a handful of frames.

Unlike run_test_light.py (which starts from raw video files), this
script reuses keyframes you ALREADY extracted for one real video
(default: L21_V029) and just copies a small slice of them into an
isolated folder. Skips the slowest step (keyframe extraction /
DINOv2 scene-change detection) entirely since it's already done.

Runs: detection -> OCR -> embeddings (dual EVA-02+CLIP-32) -> FAISS
index -> metadata sidecars, all on the tiny subset, forced onto CPU,
smallest YOLO26 variant, scene-graph disabled (Qwen 7B not feasible
here). Finishes with one real search_text() call end-to-end
(translation -> multi-stream blend) to confirm search actually
returns something sane, not just that files got written.

All outputs go to *_smoke/ folders — your real data/keyframes,
data/embeddings, data/detections, data/metadata, and the real FAISS
index/db are never touched.

IMPORTANT (Windows): pipeline/ocr.py uses multiprocessing.Pool for
OCR. On Windows this uses the 'spawn' start method, which re-imports
this file as __main__ in every worker process. Everything below
MUST stay inside `if __name__ == "__main__":`, or each OCR worker
re-runs the entire pipeline from scratch (you'll see the whole
script's output duplicated 2-4x and confusing race conditions on
the *_smoke output files).

Usage:
    python run_smoke_1video.py
    python run_smoke_1video.py --video L21_V030 --frames 12
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="L21_V029", help="Existing video name under data/keyframes/")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to copy for the test")
    args = parser.parse_args()

    SRC_VIDEO = args.video
    N_FRAMES = args.frames

    from aic26.config import settings
    from aic26.pipeline.orchestrator import PipelineOrchestrator

    # ── 1. Build the tiny keyframe subset from real, already-extracted data ─
    src_dir = settings.keyframes_dir / SRC_VIDEO
    if not src_dir.exists():
        raise SystemExit(
            f"'{src_dir}' not found. Pick a video that already has extracted "
            f"keyframes (you have L21_V029 / L21_V030 / L21_V031)."
        )

    dst_dir = Path("data/keyframes_smoke") / SRC_VIDEO
    dst_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(src_dir.glob("*.jpg"))[:N_FRAMES]
    if not frames:
        raise SystemExit(f"No .jpg frames found in '{src_dir}'.")

    for f in frames:
        shutil.copy2(f, dst_dir / f.name)

    # ── 2. Isolate every OTHER artifact so this never touches real data ─────
    settings.keyframes_dir = Path("data/keyframes_smoke")
    settings.embedding_dir = Path("data/embeddings_smoke")
    settings.clip32_embedding_dir = Path("data/clip32_embeddings_smoke")
    settings.detection_dir = Path("data/detections_smoke")
    settings.metadata_dir = Path("data/metadata_smoke")
    settings.db_path = Path("data/faiss_db_smoke.pkl")
    settings.faiss_index_path = Path("data/faiss_index_smoke.pkl")
    settings.ocr_csv_path = Path("data/ocr_results_smoke.csv")
    settings.ocr_output_path = Path("data/ocr_results_smoke.csv")  # keep both in sync, see settings.py note
    settings.device = "cpu"  # direct assignment — do NOT rely on env var precedence here
    settings.yolo_model_name = "yolo26n.pt"
    # map_keyframe_dir / media_info_dir intentionally left at their real
    # defaults — L21_V029.csv / L21_V029.json already exist for this video
    # and are keyed by video name, so reusing them is correct, not a leak.

    # An empty dir is enough — Step 1 (keyframe extraction) safely no-ops
    # when it finds zero video files, since we already have the keyframes.
    empty_video_dir = Path("data/videos_smoke_empty")
    empty_video_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Smoke test: {SRC_VIDEO}, {len(frames)} frame(s), device={settings.device}, "
          f"yolo={settings.yolo_model_name}")
    print("=" * 70)

    orchestrator = PipelineOrchestrator(config=settings)
    results = orchestrator.run(
        video_dir=str(empty_video_dir),
        skip_scene_graph=True,  # Qwen 7B not feasible on this machine
    )

    print("\n" + "=" * 70)
    print("Pipeline stats:", results.get("_pipeline_stats"))
    print("=" * 70)

    # ── 3. End-to-end search smoke test ──────────────────────────────────
    # Confirms translation + multi-stream blend actually return something,
    # not just that the offline extraction steps didn't crash.
    print("\nRunning a live search_text() call...")
    try:
        from aic26.core.retrieval import EVA02ImageRetrieval

        retriever = EVA02ImageRetrieval(
            embedding_dir=str(settings.embedding_dir),
            keyframes_dir=str(settings.keyframes_dir),
            db_path=str(settings.db_path),
            clip32_dir=str(settings.clip32_embedding_dir),
        )
        query = "một người đang đứng"  # deliberately Vietnamese, exercises translation
        hits = retriever.search_text(query, top_k=min(5, len(frames)))
        print(f"\nQuery: '{query}' -> {len(hits)} result(s)")
        for h in hits[:5]:
            print(f"  {h.get('video_name')}/{h.get('frame_idx')}  score={h.get('similarity'):.4f}")
    except Exception as exc:
        print(f"\n⚠️  Search smoke test failed: {type(exc).__name__}: {exc}")
        print("   (the offline pipeline above may still be fine — this is a separate check)")

    print("\n" + "=" * 70)
    print(
        "Done. If pipeline stats show no entries in 'steps_failed' and the "
        "search above returned results, the flow works end-to-end.\n"
        "Speed here is NOT representative of a real GPU run — this is a "
        "correctness check on a handful of CPU-processed frames, not a benchmark."
    )


if __name__ == "__main__":
    # freeze_support() is a no-op on non-frozen (non-PyInstaller) scripts,
    # but costs nothing and is the documented-safe pattern for Windows
    # multiprocessing per the Python docs referenced in the OCR error.
    multiprocessing.freeze_support()

    # Force DEVICE/YOLO_MODEL_NAME before aic26.config is ever imported —
    # anything imported at module level (including inside main(), since
    # main() itself calls `from aic26.config import settings` on first
    # invocation) reads env vars at Settings() construction time. Direct
    # assignment (not setdefault) guarantees this wins even if DEVICE=cuda
    # is already set in your shell session from earlier testing.
    os.environ["DEVICE"] = "cpu"
    os.environ["YOLO_MODEL_NAME"] = "yolo26n.pt"

    main()