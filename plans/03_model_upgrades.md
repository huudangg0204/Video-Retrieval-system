# Plan 03b: Model Upgrades v2 (Completed ✅)

> **Date**: 2026-07-02
> **Status**: COMPLETED

## Summary
Nâng cấp từ single-stream BLIP-2 lên dual-stream retrieval.

## Changes Applied
- **CLIP ViT-B-32**: Thay BLIP-2 Q-Former bằng `openai/clip-vit-base-patch32` (HuggingFace `transformers.CLIPModel`)
- **Dual-stream FAISS**: EVA-02 768d (w=0.6) + CLIP-32 512d (w=0.4) + BM25 captions (w=0.2)
- **YOLOv26x**: Thay YOLOv8x-OIV7 bằng YOLOv26x (80 COCO classes)
- **OCR Import**: Giữ EasyOCR, thêm import từ `data/raw/ocr_results.csv`
- **BM25 Caption Index**: Keyword matching trên BLIP captions

## Architecture
```
Query → EVA-02 (768d, w=0.6) ──┐
      → CLIP-32 (512d, w=0.4) ─┤→ Weighted Blend → Top-K
      → BM25 (captions, w=0.2) ┘
```

## Files Changed
- `src/aic26/core/models.py` — CLIP-32 lazy-load
- `src/aic26/core/embeddings.py` — dual-stream index
- `src/aic26/core/search.py` — 3-way blend
- `src/aic26/core/bm25.py` — BM25 caption indexing
- `src/aic26/pipeline/embedding.py` — dual embedding extraction
