# Dual-Embedding (EVA-02 + BLIP-2) — Feature Documentation

> **Status**: Production-ready on RTX 2080 Ti (11 GB VRAM)
> **Paper**: MemoriEase 3.0: A RAG-Enhanced Conversational Lifelog Retrieval System at LSC'25, §3.2
> **Date**: 2026-06-26

## 1. Overview

AIC-26 supports a **two-stream embedding pipeline** that combines two complementary vision-language models for retrieval:

| Stream | Model | Dim | Role |
|---|---|---|---|
| EVA-02 | `hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k` | 768 | Joint image+text encoder, ~427 M params, supports native text search |
| BLIP-2 | `Salesforce/blip2-itm-vit-g` (Q-Former + OPT-2.7b) | 768 | Image-conditioned embedding, ~1.2 B params, ViT-G/E + Q-Former |

Both streams store their vectors in separate FAISS `IndexFlatIP` indexes. At query time the cosine similarities from each stream are blended with a configurable weighted average (`EVA02_SEARCH_WEIGHT`, `BLIP2_SEARCH_WEIGHT`), exactly the recipe from §3.2 of the MemoriEase paper:

> *"We use a weighted average approach to combine the cosine similarity from two embeddings. The weight is chosen by retrieving several queries and tuning the best weight."*

Default weights (`EVA02_SEARCH_WEIGHT=0.6`, `BLIP2_SEARCH_WEIGHT=0.4`) were chosen by the team after offline tuning on the AIC-26 dev split.

## 2. Why two streams?

Each model captures different aspects of visual semantics:

- **EVA-02** is a CLIP-style contrastive learner. It produces tight, isotropic embeddings that are well-aligned between image and text in the *same* space — text-to-image search works directly.
- **BLIP-2 Q-Former** is an image-conditioned retrieval head trained with image-text matching. It captures fine-grained visual grounding that CLIP often smooths out, at the cost of weaker text alignment.

Blending them with a weighted average reduces the failure modes of either model used alone.

## 3. Pipeline

### 3.1 Embedding extraction

```bash
aic26 pipeline extract-dual-embeddings -i data/keyframes -b 4
```

Loops over each video folder and writes two .npy files per video (lockstep row order, L2-normalized):

```
data/embeddings/{video}.npy        # shape (N, 768), EVA-02 image features
data/blip2_embeddings/{video}.npy  # shape (N, 768), BLIP-2 Q-Former features
```

Resumable: a folder is skipped if **both** files already exist. On RTX 2080 Ti (batch=4), throughput is ~25–30 s/folder for the full 55-folder × 12,609-image corpus.

### 3.2 FAISS index build

```bash
aic26 pipeline build-dual-index
```

Stacks the per-video .npy files from both streams into single `(12609, 768)` arrays, builds two `IndexFlatIP` indexes, and pickles them together with image/video metadata:

```python
{
    'eva02_index': IndexFlatIP(768),    # 12 609 vectors
    'blip2_index': IndexFlatIP(768),    # 12 609 vectors
    'embeddings':  (12609, 768) float32,  # legacy single-stream copy
    'image_paths': ['K05_V001_converted/000000.jpg', ...],
    'video_names': ['K05_V001_converted', ...],
    'weight_eva02': 0.6,
    'weight_blip2': 0.4,
    'dual': True,
    # ... other keys preserved for compatibility
}
```

Output: `data/faiss_db.pkl` (~112 MB).

### 3.3 Search at query time

The flow is:

1. Encode the query with EVA-02 (image and text both supported).
2. Encode the query with BLIP-2 (image only; see §4 for text fallback).
3. Pull `top_k * DUAL_SEARCH_POOL_MULTIPLIER` (default `5×`) candidate rows from **each** stream's FAISS index.
4. Weighted blend: `score[i] = w_eva · sim_eva[i] + w_blip2 · sim_blip2[i]`.
5. Take the top-`k` from the blended pool.

The wider pool per stream is deliberate: if we only pulled `top_k` from each stream and blended, an image that ranks 11th on EVA-02 but 1st on BLIP-2 could be missed. The paper explicitly recommends this candidate-pool-then-blend pattern.

## 4. Important caveat: BLIP-2 text branch

The `Salesforce/blip2-itm-vit-g` checkpoint's Q-Former is **image-only**:

- `get_qformer_features(pixel_values)` accepts no text input.
- `use_qformer_text_input` is **not exposed** on `qformer_config` for this variant (only on the OPT `text_config`, which doesn't help).
- `get_text_features(input_ids, attention_mask)` exists but returns the **language-model embedding** (OPT/Flan-T5 side), not the Q-Former embedding — a different space, not directly comparable to image embeddings.

We measured the empirical effect: feeding 8 distinct text queries through the BLIP-2 path (Q-Former + 224×224 black image) produced **8 identical 768-d vectors** (pairwise cosine similarity = 1.0000). Adding this constant vector to every row of the EVA-02 stream would dominate the blend and collapse all rankings.

### 4.1 Runtime detection and fallback

To avoid this failure mode without requiring a code-level branch for every query type, `SearchEngine._blip2_text_query_usable()` runs a two-query probe at first use:

```python
a = encode_text_blip2("probe query alpha")
b = encode_text_blip2("probe query beta distinct words")
usable = bool(np.linalg.norm(a - b) > 1e-3)
```

If `usable` is `False`, `search_text` and `search_text_all` **skip** the dual blend and use the single-stream EVA-02 path. `search_image` is **unaffected** — the BLIP-2 image branch is correctly conditioned on the real query image.

If the BLIP-2 checkpoint is later swapped for a variant that accepts text input (e.g. a captioning checkpoint with `use_qformer_text_input=True`), the probe will start returning `True` and the dual blend activates for text queries automatically — no further code change required.

## 5. Configuration

All knobs live in `src/aic26/config/settings.py` and read from `.env`:

```env
USE_DUAL_EMBEDDING=true            # Master switch — when false, single-stream path
EVA02_SEARCH_WEIGHT=0.6            # Weight for EVA-02 cosine similarity in blend
BLIP2_SEARCH_WEIGHT=0.4            # Weight for BLIP-2 cosine similarity in blend
DUAL_SEARCH_POOL_MULTIPLIER=5      # top_k * this pulled from each stream before blending
BLIP2_MODEL_NAME=Salesforce/blip2-itm-vit-g
BLIP2_EMBEDDING_DIR=data/blip2_embeddings
```

`BLIP2_EMBEDDING_DIM=768` is a `Final` constant in `config/constants.py`.

`ensure_directories()` creates `data/blip2_embeddings/` on import so the dual path never fails on missing-directory errors.

## 6. CLI surface

```bash
# Offline
aic26 pipeline extract-dual-embeddings -i data/keyframes -b 4
aic26 pipeline build-dual-index

# Online (text, EVA-02 only on this checkpoint)
aic26 search text -q "a red car driving on a highway" -k 10

# API (same logic)
curl -X POST http://localhost:5000/api/search_text \
  -H "Content-Type: application/json" \
  -d '{"query":"a person riding a bicycle", "top_k":10}'
```

## 7. API & GUI wiring

- `EVA02ImageRetrieval(blip2_dir=...)` (`core/retrieval.py`) is the single facade. Setting `blip2_dir=None` falls back to single-stream behavior automatically.
- `api/server.py::_init_retriever` reads `settings.blip2_embedding_dir` and passes it through. The retriever is still lazy-loaded on first request — models do not sit in VRAM until needed.
- Both Flask routes and the CustomTkinter GUI go through the facade, so no per-interface changes were needed.
- `data/faiss_db.pkl` is backward-compatible: old single-stream pickles still load — the retriever simply skips BLIP-2 and logs that single-stream mode is active.

## 8. Performance numbers (RTX 2080 Ti 11 GB)

| Step | Throughput | Wall time |
|---|---|---|
| Dual extraction (12,609 imgs, batch=4) | ~8 imgs/s | 25 min |
| Dual index build | n/a | <1 s |
| Search text (single-stream fallback) | — | ~60 ms / query |
| Search image (full dual blend, top-25 pool) | — | ~200 ms / query |

Peak VRAM with both models resident (EVA-02 + BLIP-2): **~9.6 GB**. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation if you see OOM on a tighter budget.

## 9. Files changed for this feature

- `src/aic26/config/settings.py` — added 5 fields, `ensure_directories()`
- `src/aic26/config/constants.py` — added `BLIP2_EMBEDDING_DIM`
- `src/aic26/core/models.py` — `_load_blip2`, `_pool_blip2_query`, `encode_image_blip2`, `encode_text_blip2`, `encode_images_blip2_batch`
- `src/aic26/pipeline/embedding.py` — `process_images_dual`, `_embed_image_folder_dual`, `_flush_dual_batch`
- `src/aic26/pipeline/indexer.py` — `build_dual`, `_load_with_paths`
- `src/aic26/core/embeddings.py` — `EmbeddingIndex(blip2_dir=…)`, `encode_text_blip2`, `encode_image_blip2`, `get_embedding_blip2`
- `src/aic26/core/search.py` — `SearchEngine(weight_eva02, weight_blip2, pool_multiplier)`, `_dual_topk_indices`, `_blip2_text_query_usable` probe
- `src/aic26/core/retrieval.py` — facade passes `blip2_dir` and weights through
- `src/aic26/api/server.py` — passes `blip2_dir` to retriever
- `src/aic26/cli/main.py` — `pipeline extract-dual-embeddings`, `pipeline build-dual-index`
- `src/aic26/pipeline/orchestrator.py` — routes Step 4 (embeddings) and Step 5 (index) to dual path when `use_dual_embedding=True`

## 10. Verification commands

```bash
# Quick sanity (assumes data/ already populated)
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from aic26.config import settings
print('dual:', settings.use_dual_embedding)
print('weights:', settings.eva02_search_weight, settings.blip2_search_weight)
print('blip2 dir:', settings.blip2_embedding_dir)
"
ls data/embeddings/*.npy | wc -l            # expect 31
ls data/blip2_embeddings/*.npy | wc -l      # expect 31
ls -lh data/faiss_db.pkl                    # ~112 MB

# End-to-end via API (server must be running)
curl -X POST http://localhost:5000/api/search_text \
  -H "Content-Type: application/json" \
  -d '{"query":"a red car driving on a highway","top_k":5}'
```
