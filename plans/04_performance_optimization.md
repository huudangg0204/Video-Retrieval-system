# Plan 04: Performance Optimization (Completed ✅)

> **Date**: 2026-07-02
> **Status**: COMPLETED

## Summary

Tối ưu tốc độ search và VRAM usage trên RTX 3050 (4GB).

## Changes Applied

### PA1: Lazy-load Models
- `models.py`: Faster R-CNN + BLIP ITM → lazy `@property`
- Chỉ EVA-02 CLIP eager-load (~1.7GB VRAM)
- VRAM tiết kiệm: ~1.4GB

### PA2: Batch BLIP Reranking
- `rerank.py`: Batch inference (batch_size=8)
- Speedup: ~4-8x cho TRAKE reranking

### PA3: Reduce Rerank Candidates
- `rerank.py`: max_candidates=20 (mặc định)
- Speedup: reranking ~5x nhanh hơn

### PA4: Cache Metadata In-Memory
- `api/routes/search.py`: Preload tất cả JSON sidecars vào dict (~3MB RAM)
- Speedup: metadata lookup ~100x (disk I/O → dict access)

### PA5: Precomputed Detections
- `search.py`: Object filter dùng YOLOv26 detections từ metadata thay vì Faster R-CNN realtime
- VRAM tiết kiệm: ~170MB

### PA6: Flask Threaded Mode
- `cli/main.py`: `threaded=True` → UI không bị đứng khi search

## Results

| Metric | Before | After |
|--------|--------|-------|
| VRAM | 3925 MiB | 2489 MiB (-36%) |
| Search time | ~94s | ~0.1s (-99.9%) |
| Startup time | ~20s | ~10s (-50%) |
| Metadata lookup | ~1s (100 files) | ~0ms (cache) |

## Files Changed
- `src/aic26/core/models.py`
- `src/aic26/core/search.py`
- `src/aic26/core/rerank.py`
- `src/aic26/api/routes/search.py`
- `src/aic26/cli/main.py`
