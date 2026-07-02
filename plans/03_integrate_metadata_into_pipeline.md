# Plan 03a: Integrate Metadata into Pipeline (Completed ✅)

> **Date**: 2026-07-02
> **Status**: COMPLETED

## Summary
- Metadata extraction chạy mặc định trong `run-all` (opt-out via `--skip-metadata`)
- Tái sử dụng detection JSON từ Step 2 + OCR CSV từ Step 3 — không chạy lại
- Auto-merge OCR text vào sidecars (sub-step 6b)
- Caption merge via BLIP (sub-step 6c)
- GPU cleanup (`_cleanup_gpu()`) giữa mỗi step cho 4GB VRAM safety

## Pipeline Steps (Sequential, GPU-safe)
1. Keyframes (DINOv2) → cleanup
2. Detection (YOLOv26x, num_workers=1) → cleanup
3. Embeddings (EVA-02 768d + CLIP-32 512d) → cleanup
4. FAISS Index Build → cleanup
5. Metadata Extract (merge detections + OCR + captions)

## Files Changed
- `src/aic26/pipeline/orchestrator.py` — added `_cleanup_gpu()`, sequential step execution
- `src/aic26/pipeline/metadata.py` — OCR merge with stem mapping (0000.jpg → 001)
- `src/aic26/pipeline/detection.py` — `num_workers=1` default
- `src/aic26/cli/main.py` — pipeline CLI commands
