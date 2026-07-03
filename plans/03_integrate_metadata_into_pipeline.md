# Plan 03a: Integrate Metadata into Pipeline (Completed ✅)

> **Date**: 2026-07-02 → 2026-07-03
> **Status**: COMPLETED (Full-scale run done)

## Summary
- Metadata extraction chạy mặc định trong `run-all` (opt-out via `--skip-metadata`)
- Tái sử dụng detection JSON từ Step 2 + OCR CSV từ Step 3 — không chạy lại
- Auto-merge OCR text vào sidecars (sub-step 6b)
- Caption merge via BLIP (sub-step 6c)
- GPU cleanup (`_cleanup_gpu()`) giữa mỗi step cho 4GB VRAM safety

## Pipeline Steps (Sequential, GPU-safe)
1. Video Conversion (ffmpeg) → cleanup
2. Keyframe Extraction (DINOv2) → cleanup
3. Object Detection (YOLOv26x, num_workers=1) → cleanup
4. OCR Extraction (EasyOCR) → cleanup
5. Embedding Extraction (EVA-02 768d + CLIP-32 512d) → cleanup
6. FAISS Dual Index Build → cleanup
7. Metadata Extract (merge detections + OCR + captions + Places365)

## Full-Scale Results (2026-07-03)
- **93 video folders** processed (K11, K12, K13 series)
- **41,675 keyframes** extracted and indexed
- **41,675 metadata JSON sidecars** created
- **944 frames** with OCR text detected
- **Total runtime**: ~14.5 hours (RTX 3050 4GB)
- **FAISS index**: 326.7 MB (41,675 vectors × 2 streams)

## Files Changed
- `src/aic26/pipeline/orchestrator.py` — `_cleanup_gpu()`, sequential step execution
- `src/aic26/pipeline/metadata.py` — OCR merge with stem mapping (0000.jpg → 001)
- `src/aic26/pipeline/detection.py` — `num_workers=1` default, skip if JSON exists
- `src/aic26/cli/main.py` — pipeline CLI commands
- `run_remaining_videos.py` — [NEW] Script to run full pipeline on remaining folders
