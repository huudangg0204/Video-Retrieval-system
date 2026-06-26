# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AIC-26 — a multilingual (Vietnamese / English) multimodal video retrieval system for the AI Challenge competition. It indexes video keyframes with EVA-02 CLIP embeddings, stores them in a FAISS vector index, and serves text/image/temporal queries through a Flask REST API, a CustomTkinter desktop GUI, and a Click CLI.

Source layout (under `src/aic26/`, built with Hatchling):

| Module | Purpose |
|---|---|
| `core/` | Retrieval engine — the **only** thing interfaces should call. `EVA02ImageRetrieval` (`core/retrieval.py`) is the public facade. Sub-modules: `models.py` (lazy-loaded EVA-02 CLIP + Faster R-CNN + BLIP ITM, **BLIP-2 Q-Former loaded on demand** for dual mode), `embeddings.py` (in-memory + on-disk embedding index, O(1) lookup; loads a second BLIP-2 stream when present), `search.py` (text/image search; weighted-average blend in dual mode), `trake.py` (temporal DP), `rerank.py` (BLIP ITM), `database.py` (persistent query store, pickle), `export.py` (CSV submissions). |
| `pipeline/` | Offline data preparation. `orchestrator.py` runs the 5 sequential steps; individual modules `keyframe.py` (DINOv2 scene-change detection), `detection.py` (YOLOv8 OIV7), `ocr.py` (EasyOCR), `embedding.py` (EVA-02 batches), `indexer.py` (FAISS IndexFlatIP). |
| `api/` | Flask app factory (`server.py`) + blueprints in `routes/` (`search`, `trake`, `export`, `video`, `system`) and `middleware/error_handler.py`. Lazy-loads the retriever on first request. |
| `gui/` | CustomTkinter desktop apps — `retrieval_app.py` (main search UI) and `filter_app.py` (object-filter UI). |
| `cli/` | Click entry point. `main.py` defines the `aic26` console script with subcommands `pipeline`, `search`, `serve`, `config`, `check`, `download`. |
| `config/` | `Settings` (Pydantic, reads `.env`) and `constants.py` (model dims, FAISS index names, COCO labels — `EVA02_EMBEDDING_DIM = 768`). The `settings` singleton auto-creates all data directories on import. |
| `utils/` | `logging` (loguru setup), `frame_utils`, `media_utils` (FPS / frame-index CSV lookup, `frame_to_timestr`), `file_utils`, `video_converter` (AV1/MKV → MP4 H.264). |

## Development setup

Install PyTorch **first** for the right CUDA build (the package's pip metadata is empty), then the project:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # or /cpu
pip install -e ".[dev]"          # pulls pytest, ruff, mypy, pre-commit
pre-commit install
```

`yolov8x-oiv7.pt` is the default detection weights and lives at the repo root; `oidv7-class-descriptions-boxable.csv` is the class mapping. The shipped `.env` overrides defaults — `.env.example` lists every variable.

## Common commands

```bash
# CLI entry point — runs every subsystem
aic26 --help
aic26 check                                 # verify deps, GPU, paths
aic26 config --show                         # dump active Settings

# Pipeline (each step is independently runnable)
aic26 pipeline run-all --video-dir data/videos [--skip-detection --skip-ocr --skip-embeddings --skip-index]
aic26 pipeline extract-keyframes -i data/videos -o data/keyframes
aic26 pipeline extract-embeddings -i data/keyframes -b 40   # GPU strongly recommended
aic26 pipeline build-index -i data/embeddings --index-type flat_ip

# Serve
aic26 serve api --host 0.0.0.0 --port 5000   # Flask at /api, web UI at /
aic26 serve gui                              # CustomTkinter search app
aic26 serve filter                           # object-filter GUI

# Search
aic26 search text -q "a red car" -k 20 -o "car,person" -t 0.5
```

Docker (cross-platform sync for the team):

```bash
docker compose -f docker/docker-compose.yml --profile pipeline up --build   # full ETL with GPU
docker compose -f docker/docker-compose.yml --profile gpu up --build -d      # API server with GPU
```

## Testing

Pytest is configured in `pyproject.toml`. Coverage runs by default; markers `slow`, `gpu`, `integration` exist for deselection.

```bash
pytest                                          # full suite w/ coverage report
pytest -m "not slow and not gpu"                # CI-style subset
pytest tests/unit/test_core.py -k search_text   # single test by node id
pytest --cov=aic26 --cov-report=html            # HTML coverage in htmlcov/
```

Tests live in `tests/unit/`; `tests/conftest.py` provides `temp_dir`, `test_settings`, `sample_embeddings`, `sample_results`, `sample_video_names` fixtures.

CI (`.github/workflows/ci.yml`) runs on push/PR to `main` and `develop`: lint → test on Python 3.10/3.11 → Docker build (push only) → Trivy scan.

## Lint / type / format

`ruff` handles lint + format (Black-compatible, line-length 100), `mypy` runs in strict mode but ignores missing imports for torch/transformers/ultralytics/etc. (see `[tool.mypy.overrides]` in `pyproject.toml`).

```bash
ruff check --fix src/ tests/
ruff format src/ tests/
mypy src/ --ignore-missing-imports
pre-commit run --all-files     # mirrors what CI runs
```

## Key conventions and gotchas

- **Single entry point**: every interface (API routes, GUI, CLI search commands) instantiates `EVA02ImageRetrieval` and forwards to it. Don't reimplement search logic — add to `core/` and expose via the facade.
- **Lazy model loading**: `ModelRegistry` loads EVA-02 + Faster R-CNN + BLIP ITM in `__init__`. `api/server.py` wraps it in a `_get_retriever` closure so models load on first request, not on import.
- **Settings auto-mkdir**: importing `from aic26.config import settings` creates every data directory. Tests that point at a temp dir must pass `data_dir=...` explicitly so the real `data/` is never polluted (see `test_settings` fixture).
- **Search-API auto-bootstrap**: `routes/search.py::_ensure_embeddings_loaded` will kick off the full pipeline if `data/embeddings/*.npy` is empty *and* `data/videos/` is non-empty. The call resets `AIC26_RETRIEVER` so the next request picks up fresh embeddings.
- **TRAKE strategies**: `TrakeEngine.trake_closest` minimizes frame gaps (greedy DP); `trake_highest` maximizes summed similarity (option `0` standard, `1` threshold-gated). The API exposes them as `option="Closest"` / `option="Highest"`.
- **Object filter input**: `search_text` accepts `objects` (comma-separated COCO category names) + `threshold`; detection is done on the query image, not pre-computed per keyframe.
- **EVA-02 rerank cue**: an event string ending in `*` (e.g. `"a player shooting*"`) triggers BLIP ITM reranking — keep that suffix convention when wiring new features.
- **Dual-stream search** (MemoriEase 3.0): when `USE_DUAL_EMBEDDING=true` and `data/blip2_embeddings/` is populated, `EmbeddingIndex` loads a second `blip2_embeddings` array and `SearchEngine._dual_topk_indices` blends `EVA02_SEARCH_WEIGHT * sim_eva02 + BLIP2_SEARCH_WEIGHT * sim_blip2` over a `top_k * DUAL_SEARCH_POOL_MULTIPLIER` candidate pool. Two `IndexFlatIP` indexes are built by `IndexBuilder.build_dual` and pickled together. BLIP-2 text queries go through Q-Former with a 224×224 black image (its `get_qformer_features` is image-only).
- **BLIP-2 VRAM**: ~5 GB on RTX 2080 Ti. Lazy-loaded — `ModelRegistry._load_blip2()` runs on first encode, not at startup. Pre-warm it before a hot search session.
- **CLI has a duplicate `serve api` command** in `src/aic26/cli/main.py` (one is the Phase-4 placeholder, the other is the real implementation). Click uses the last definition, so the working one wins — but be aware if you grep.
- **Encoding**: `core/retrieval.py` reconfigures `sys.stdout` to UTF-8 for Windows compatibility.
- **DB**: `VectorDatabase` (`core/database.py`) pickles to `data/faiss_db.pkl`; query history endpoints (`/api/stored_queries`, `/api/reload_query/<idx>`) read from it.
- **Constants** in `config/constants.py` are `Final` and intentionally not user-tunable — model output dims, FAISS index name mapping, OCR language list, submission row/gap defaults. Add new knobs to `Settings` instead.

## Environment variables (subset)

Pydantic `Settings` reads `.env` (case-insensitive). The most useful knobs:

- `DEVICE` (default `cuda`; falls back to CPU if unavailable)
- `DEFAULT_TOP_K=100`, `DEFAULT_SCORE_THRESHOLD=0.2`, `DEFAULT_OBJ_THRESHOLD=0.5`, `RERANK_THRESHOLD=0.4`
- `HOST=0.0.0.0`, `PORT=5000`, `DEBUG=false`, `FLASK_SECRET_KEY`, `MAX_UPLOAD_SIZE_MB=16`
- `MAX_WORKERS=4`, `KEYFRAME_SIMILARITY_THRESHOLD=0.8`, `KEYFRAME_AUTO_ACCEPT_SECONDS=10.0`
- `LOG_LEVEL=INFO`, `LOG_FILE` (optional file path)
- `EVA02_MODEL_NAME`, `EVA02_PRETRAINED`, `BLIP_MODEL_NAME`, `YOLO_MODEL_NAME`, `OID_MAPPING_CSV`
- `TRAKE_DEFAULT_TOP_K=200`, `TRAKE_DEFAULT_CANDIDATES=200`, `GAP_KEYFRAME=15`

`aic26 config --show` prints the resolved values.

## Documentation

`mkdocs.yml` configures mkdocs-material with `mkdocstrings` autodoc. Source pages are referenced via nav; build with `pip install -e ".[docs]" && mkdocs serve`.