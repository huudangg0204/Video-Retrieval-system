# Plan 03b: Model Upgrades v2 (Completed ✅)

> **Date**: 2026-07-02
> **Status**: COMPLETED

## Summary
Nâng cấp từ single-stream BLIP-2 lên dual-stream retrieval + BM25 keyword search.

## Changes Applied
- **CLIP ViT-B-32**: Thay BLIP-2 Q-Former bằng `openai/clip-vit-base-patch32` (HuggingFace `transformers.CLIPModel`)
- **Dual-stream FAISS**: EVA-02 768d (w=0.6) + CLIP-32 512d (w=0.4) + BM25 captions (w=0.2)
- **YOLOv26x**: Thay YOLOv8x-OIV7 bằng YOLOv26x (80 COCO classes)
- **OCR Import**: Giữ EasyOCR, thêm import từ `data/raw/ocr_results.csv`
- **BM25 Caption Index**: Keyword matching trên BLIP captions
- **CLIP-32 Eager-load**: Load khi server khởi động thay vì lazy-load (PA9)

## Architecture
```
Query → EVA-02 (768d, w=0.6) ──┐
      → CLIP-32 (512d, w=0.4) ─┤→ Weighted Blend → Top-K
      → BM25 (captions, w=0.2) ┘
```

## Serving Models (tại server)
| Model | VRAM | Load Strategy |
|---|---|---|
| EVA-02 CLIP (427.8M params) | ~1.7 GB | **Eager** — always loaded |
| CLIP ViT-B-32 (512-dim) | ~0.5 GB | **Eager** — loaded at startup |
| BLIP ITM (reranker) | ~0.9 GB | **Lazy** — only for TRAKE |
| Faster R-CNN | ~0.2 GB | **Lazy** — fallback if no metadata |

## Files Changed
- `src/aic26/core/models.py` — CLIP-32 eager-load (was lazy)
- `src/aic26/core/embeddings.py` — dual-stream index
- `src/aic26/core/search.py` — 3-way blend (EVA-02 + CLIP-32 + BM25)
- `src/aic26/core/bm25.py` — BM25 caption indexing (reads from pickle cache)
- `src/aic26/pipeline/embedding.py` — dual embedding extraction
