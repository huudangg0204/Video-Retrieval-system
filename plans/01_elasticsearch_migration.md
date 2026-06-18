# Plan: Elasticsearch Migration (MemoriEase 3.0)

> Created: 2026-06-18  
> Status: **COMPLETED**  
> Branch: `feature/elasticsearch-migration`

## Mục Tiêu

Chuyển hệ thống retrieval từ FAISS in-memory sang Elasticsearch 8.13 làm backend chính, giữ local machine làm processing/API server.

## Architecture

```
Local Machine (Processing Server)
    ├── EVA-02 (768-dim) embedding
    ├── BLIP-ITM (768-dim) embedding  ← dual embedding (Option C Lite)
    └── API / GUI

Elasticsearch (Docker)
    ├── Index: aic26_keyframes
    │   ├── eva02_embedding (dense_vector 768d, kNN)
    │   ├── blip_embedding  (dense_vector 768d, kNN)
    │   ├── ocr_text        (full-text search)
    │   └── detected_objects (keyword filter)
    └── Index: aic26_queries (query history)
```

**Dual Embedding Weights (RTX 3050 4GB constraint):**
- EVA-02: 60% weight
- BLIP-ITM: 40% weight

## Files Changed

### Infrastructure
- `elasticsearch_client.py` [NEW] — ES wrapper, dual kNN, OCR search, query history
- `config/settings.py` — +13 ES settings
- `docker/docker-compose.yml` — ES 8.13 + Kibana services
- `pyproject.toml` — `elasticsearch>=8.0.0,<9.0.0`
- `.env.example` — ES env vars

### Models & Encoding
- `core/models.py` — `encode_image_blip()`, `encode_text_blip()`
- `core/embeddings.py` — Refactor → `EmbeddingEncoder` thin wrapper

### Pipeline
- `pipeline/indexer.py` — `ElasticsearchIndexer` (bulk index .npy → ES)

### Search Engine
- `core/search.py` — ES kNN dual-embedding search
- `core/database.py` — `QueryHistoryStore` (ES-backed, replaces pickle)
- `core/retrieval.py` — Updated facade, connects to ES

### CLI
- `cli/main.py` — `check-es`, `pipeline init-es-index`, `pipeline index-to-es`

## CLI Usage

```bash
# Start Elasticsearch
docker compose -f docker/docker-compose.yml up elasticsearch -d

# Create index
py -3.10 -m aic26.cli.main pipeline init-es-index

# Index data
py -3.10 -m aic26.cli.main pipeline index-to-es --embedding-dir data/embeddings

# Search
py -3.10 -m aic26.cli.main search text --query "a person riding a bicycle"

# Status check
py -3.10 -m aic26.cli.main check-es
```

## Known Issues Fixed
- `elasticsearch==9.x` incompatible with ES 8.13 → pinned `<9.0.0`
- Windows `UnicodeEncodeError` with emoji in CLI → replaced with ASCII
- `aic26` not in PATH → `pip install -e .` + add Scripts to User PATH
