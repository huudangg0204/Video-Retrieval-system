# Plan 04: Performance Optimization (Completed ✅)

> **Date**: 2026-07-02 → 2026-07-03
> **Status**: COMPLETED (Phase 2 applied)

## Summary

Tối ưu tốc độ search, VRAM usage và thời gian khởi động server trên RTX 3050 (4GB).

## Phase 1 — In-Session Optimizations (2026-07-02) ✅

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
- `api/routes/search.py`: Preload tất cả JSON sidecars vào dict
- Speedup: metadata lookup ~100x (disk I/O → dict access)

### PA5: Precomputed Detections
- `search.py`: Object filter dùng YOLOv26 detections từ metadata thay vì Faster R-CNN realtime
- VRAM tiết kiệm: ~170MB

### PA6: Flask Threaded Mode
- `cli/main.py`: `threaded=True` → UI không bị đứng khi search

## Phase 2 — Full-Scale Startup Optimizations (2026-07-03) ✅

Khi scale từ 1,360 → 41,675 keyframes, Phase 1 bộc lộ bottleneck mới:

### PA7: Pre-built Metadata Cache (Pickle)
- **Vấn đề**: BM25 đọc 41,675 file JSON riêng lẻ → mất ~5 phút khởi động
- **Giải pháp**: `scripts/build_meta_cache.py` serialize toàn bộ metadata vào 1 file `_cache.pkl` (~48 MB)
- **Kết quả**: BM25 load từ pickle chỉ **< 1 giây** (giảm ~300x)

### PA8: Shared Metadata Cache (BM25 ↔ API Route)
- **Vấn đề**: API route `_get_meta_cache()` đọc lại 41,675 JSON files lần thứ 2
- **Giải pháp**: API route reuse `bm25_index._meta_cache` trực tiếp (zero copy)
- **Kết quả**: Metadata attach tốn **0 giây** thay vì ~5 phút

### PA9: Eager-load CLIP-32
- **Vấn đề**: CLIP-32 lazy-load gây delay ~8 giây ở request đầu tiên
- **Giải pháp**: `models.py.__init__()` gọi `_load_clip32()` ngay khi `use_dual_embedding=True`
- **Kết quả**: Query đầu tiên nhanh ngay, không có cold-start delay

## Results (Phase 2, 41,675 keyframes)

| Metric | Before (Phase 1) | After (Phase 2) |
|--------|-------------------|------------------|
| Server startup | ~5-7 phút | **~22 giây** (-95%) |
| First query latency | ~8s (CLIP-32 lazy) | **~0.1s** (eager) |
| BM25 metadata load | ~5 phút (41K JSONs) | **< 1s** (pickle) |
| Metadata attach (API) | ~5 phút (41K JSONs) | **0s** (shared cache) |
| VRAM usage (idle) | ~2.5GB | ~2.5GB (unchanged) |
| Search time (top_k=100) | ~0.1s | ~0.1s (unchanged) |

## Files Changed

### Phase 1
- `src/aic26/core/models.py`
- `src/aic26/core/search.py`
- `src/aic26/core/rerank.py`
- `src/aic26/api/routes/search.py`
- `src/aic26/cli/main.py`

### Phase 2
- `scripts/build_meta_cache.py` — [NEW] Pre-build pickle cache
- `src/aic26/core/bm25.py` — Load from `_cache.pkl` instead of 41K JSON files
- `src/aic26/core/models.py` — Eager-load CLIP-32 at startup
- `src/aic26/api/routes/search.py` — Reuse BM25's shared metadata cache
