# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AIC-26 — a multilingual (Vietnamese / English) multimodal video retrieval system for the AI Challenge competition. It indexes video keyframes with EVA-02 CLIP embeddings, stores them in a FAISS vector index, and serves text/image/temporal queries through a Flask REST API, a CustomTkinter desktop GUI, and a Click CLI.

In addition to EVA-02 retrieval, the project extracts **per-keyframe metadata sidecars** (object bbox positions, Places365 scene labels, and Qwen2.5-VL scene-graph triples) — see [Metadata extraction](#metadata-extraction) below. This is a separate feature on `feature/create-metadata` and is still being filled out per `docs/plans/metadata-improvement-plan.md`.

## Architecture

Source layout (under `src/aic26/`, built with Hatchling):

| Module | Purpose |
|---|---|
| `core/` | Retrieval engine — the **only** thing interfaces should call. `EVA02ImageRetrieval` (`core/retrieval.py`) is the public facade. Sub-modules: `models.py` (lazy-loaded EVA-02 CLIP + Faster R-CNN + BLIP ITM, **BLIP-2 Q-Former loaded on demand** for dual mode), `embeddings.py` (in-memory + on-disk embedding index, O(1) lookup; loads a second BLIP-2 stream when present), `search.py` (text/image search; weighted-average blend in dual mode), `trake.py` (temporal DP), `rerank.py` (BLIP ITM), `database.py` (persistent query store, pickle), `export.py` (CSV submissions). |
| `pipeline/` | Offline data preparation. `orchestrator.py` runs the 6 sequential steps; individual modules `keyframe.py` (DINOv2 scene-change detection), `detection.py` (YOLOv8 OIV7 + 3×3 grid `position` field), `ocr.py` (EasyOCR), `embedding.py` (EVA-02 batches), `indexer.py` (FAISS IndexFlatIP), plus the **metadata stack** (`position.py`, `places365.py`, `scene_graph.py`, `metadata.py` — see below). |
| `api/` | Flask app factory (`server.py`) + blueprints in `routes/` (`search`, `trake`, `export`, `video`, `system`) and `middleware/error_handler.py`. Lazy-loads the retriever on first request. |
| `gui/` | CustomTkinter desktop apps — `retrieval_app.py` (main search UI) and `filter_app.py` (object-filter UI). |
| `cli/` | Click entry point. `main.py` defines the `aic26` console script with subcommands `pipeline`, `search`, `serve`, `config`, `check`, `download`. The `pipeline` group includes `extract-metadata`, `verify-metadata`, `metadata-stats` for the metadata feature. |
| `config/` | `Settings` (Pydantic, reads `.env`) and `constants.py` (model dims, FAISS index names, COCO labels, scene-grid constants — `EVA02_EMBEDDING_DIM = 768`). The `settings` singleton auto-creates all data directories on import. |
| `utils/` | `logging` (loguru setup), `frame_utils`, `media_utils` (FPS / frame-index CSV lookup, `frame_to_timestr`), `file_utils`, `video_converter` (AV1/MKV → MP4 H.264). |

## Development setup

Install PyTorch **first** for the right CUDA build (the package's pip metadata is empty), then the project:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # or /cpu
pip install -e ".[dev]"          # pulls pytest, ruff, mypy, pre-commit
pre-commit install
```

Weights shipped at the repo root:

- `yolov8x-oiv7.pt` — YOLOv8x Open Images V7 detector (default `YOLO_MODEL_NAME`).
- `oidv7-class-descriptions-boxable.csv` — 601-row MID ↔ entity mapping tracked in git; without this `pipeline/detection.py` silently emits `detections: []`.
- `models/places365/` (gitignored) — `resnet50_places365.pth.tar` + `categories_places365.txt` produced by `scripts/download_places365.py`.

The shipped `.env` overrides defaults — `.env.example` lists every variable.

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

# Metadata extraction (see section below)
aic26 pipeline extract-metadata -i data/keyframes [--skip-detections --skip-scene --skip-scene-graph]
aic26 pipeline verify-metadata --video K05_V004 --frame-idx 42
aic26 pipeline metadata-stats -i data/metadata
python scripts/download_places365.py        # one-time: fetch Places365 weights
python scripts/smoke_test_metadata.py --video-dir data/keyframes/K05_V004 --limit 50
python scripts/smoke_test_metadata.py --video-dir data/keyframes/K05_V004 --real-models --limit 50  # GPU

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

## Metadata extraction

Lives on branch `feature/create-metadata`; design docs in `docs/specs/metadata-extraction.md` and `docs/plans/metadata-improvement-plan.md`. Goal: enrich every keyframe with three orthogonal signals so the retriever can answer position- and relation-aware queries without changing the EVA-02 embedding pipeline.

### Sidecar schema (v1, per `docs/plans/metadata-improvement-plan.md` §3)

Written to `data/metadata/{video}/{frame_idx}.json`:

```json
{
  "video": "K05_V004",
  "frame_idx": 42,
  "detections":  [{"label": "Person", "score": 0.92, "bbox": [10,20,100,200], "position": "top-left"}],
  "scene":       {"category": "airport_terminal", "confidence": 0.633,
                  "top5": [["airport_terminal", 0.633], ["indoor", 0.185]]},
  "scene_graph": [{"s": "person", "p": "hold", "o": "cup", "confidence": 0.9}]
}
```

Semantic rules: `detections` is always `[]` (cheap pass); `scene`/`scene_graph` are `null` when **skipped**, `[]`/`{...}` when ran-but-empty. Backward-compatible.

### Module map

| File | Role |
|---|---|
| `pipeline/position.py` | `position_from_bbox(bbox, w, h) -> "top-left" \| ... \| None` — 3×3 grid; bboxes < 5% of image size return `None` (per spec R5). |
| `pipeline/places365.py` | `Places365Classifier` + `Places365Classifier.from_weights(weights, labels, device, top_k)` factory that builds the torchvision ResNet50 head from a CSAIL `.pth.tar`, strips DataParallel `module.` prefixes, and tolerates FC shape mismatches (`strict=False`). Also exposes `_infer_scene_group(label)` and de-duplicates repeated leaf names (`/p/platform/train_station_platform` → `train_station_platform`). |
| `pipeline/detection.py` | YOLOv8 OIV7 wrapper. Skips the `LabelName,DisplayName` header row when parsing `oidv7-class-descriptions-boxable.csv`; calls `self._load_model()` lazily inside `_detect_single()` so the weights load on first inference (not just via `ObjectDetector.process(...)`). Output dict includes `detection_positions: list[str]`. |
| `pipeline/scene_graph.py` | `extract_scene_graph(image, model, max_triples, max_new_tokens) -> list[(s,p,o)]` — prompts Qwen2.5-VL-7B-Instruct (4-bit via `bitsandbytes`), parses the JSON block out of free-form output (`_extract_first_json_object`), normalizes triples via `normalize_triple`. |
| `pipeline/metadata.py` | `MetadataExtractor` orchestrator + `KeyframeMetadata`/`MetadataStats` dataclasses + factory functions `load_precomputed_detections(detection_dir)` (reads existing `data/detections/{video}/{frame}.json` files — preferred for production) and `build_metadata_callables(...)` (loads Places365 + scene-graph from settings). Also `aggregate_metadata_stats(metadata_root)` for `metadata-stats` and `apply_temporal_smoothing(...)` (Phase 2 / plan task 2.3). |
| `pipeline/orchestrator.py` | `PipelineOrchestrator.run()` gained `extract_metadata: bool = False` (per AD-5) — opt-in via the `--extract-metadata` flag on `aic26 pipeline run-all`. Step 6 is wired to call `MetadataExtractor.process_all`. **Currently passes dummy lambdas** — see Outstanding #2 in `docs/plans/metadata-improvement-plan.md` (Task 1.2). |
| `cli/main.py` | `aic26 pipeline extract-metadata` (with `--skip-detections`, `--skip-scene`, `--skip-scene-graph`, `--overwrite`), `verify-metadata`, `metadata-stats`. |

### Quick smoke + verification

```bash
# 1. (one time) Get Places365 weights — streamed from places2.csail.mit.edu + raw GitHub
python scripts/download_places365.py
#  → writes models/places365/{resnet50_places365.pth.tar, categories_places365.txt}

# 2. Run the smoke (defaults to stub models — no GPU needed)
python scripts/smoke_test_metadata.py --video-dir data/keyframes/K05_V004 --limit 50

# 3. Or with real models (YOLO + Places365, Qwen stubbed None for MVP)
python scripts/smoke_test_metadata.py --video-dir data/keyframes/K05_V004 --real-models --limit 50

# 4. Aggregate
aic26 pipeline metadata-stats -i data/metadata
#   → sidecars, top scene categories, scene-graph density

# 5. Inspect one sidecar
cat data/metadata/K05_V004/001.json | jq .
```

Exit codes: `0` ok, `1` bad args / missing dir, `2` partial failure (some keyframes failed). Re-runs are idempotent: existing sidecars are skipped unless `--overwrite` is given.

## Testing

Pytest is configured in `pyproject.toml`. Coverage runs by default; markers `slow`, `gpu`, `integration` exist for deselection. The default `addopts` enable coverage **and a coverage-rate assertion** (`--cov-fail-under=0`) — passing 0 means coverage failure doesn't break the run.

```bash
pytest                                          # full suite w/ coverage report
pytest -m "not slow and not gpu"                # CI-style subset
pytest tests/unit/test_metadata.py -k process_video  # single test by node id
pytest tests/unit/test_places365_loader.py -v   # loader-only
pytest --cov=aic26 --cov-report=html            # HTML coverage in htmlcov/
```

Tests live in `tests/unit/`; `tests/conftest.py` provides `temp_dir`, `test_settings`, `sample_embeddings`, `sample_results`, `sample_video_names` fixtures. The metadata feature adds `test_metadata.py`, `test_metadata_constants.py`, `test_position.py`, `test_detection_position.py`, `test_places365.py`, `test_places365_loader.py`, `test_scene_graph.py`, `test_cli_metadata.py`, `test_smoke_metadata.py`, `test_download_places365.py`.

CI (`.github/workflows/ci.yml`) runs on push/PR to `main` and `develop`: lint → test on Python 3.10/3.11 → Docker build (push only) → Trivy scan.

## Lint / type / format

`ruff` handles lint + format (Black-compatible, line-length 100), `mypy` runs in strict mode but ignores missing imports for torch/transformers/ultralytics/etc. (see `[tool.mypy.overrides]` in `pyproject.toml`). `pre-commit` runs `ruff` + format checks.

```bash
ruff check --fix src/ tests/
ruff format src/ tests/
mypy src/ --ignore-missing-imports
pre-commit run --all-files     # mirrors what CI runs
```

## Key conventions and gotchas

- **Single entry point**: every interface (API routes, GUI, CLI search commands) instantiates `EVA02ImageRetrieval` and forwards to it. Don't reimplement search logic — add to `core/` and expose via the facade.
- **Lazy model loading**: `ModelRegistry` loads EVA-02 + Faster R-CNN + BLIP ITM in `__init__`. `api/server.py` wraps it in a `_get_retriever` closure so models load on first request, not on import. Same pattern in the new metadata feature: `Places365Classifier.__init__` takes an optional pre-loaded `model`, and `from_weights()` is the canonical way to build one from a `.pth.tar`. `ObjectDetector._detect_single()` now calls `self._load_model()` internally so callers don't need to know.
- **Settings auto-mkdir**: importing `from aic26.config import settings` creates every data directory. Tests that point at a temp dir must pass `data_dir=...` explicitly so the real `data/` is never polluted (see `test_settings` fixture).
- **Search-API auto-bootstrap**: `routes/search.py::_ensure_embeddings_loaded` will kick off the full pipeline if `data/embeddings/*.npy` is empty *and* `data/videos/` is non-empty. The call resets `AIC26_RETRIEVER` so the next request picks up fresh embeddings.
- **TRAKE strategies**: `TrakeEngine.trake_closest` minimizes frame gaps (greedy DP); `trake_highest` maximizes summed similarity (option `0` standard, `1` threshold-gated). The API exposes them as `option="Closest"` / `option="Highest"`.
- **Object filter input**: `search_text` accepts `objects` (comma-separated COCO category names) + `threshold`; detection is done on the query image, not pre-computed per keyframe.
- **EVA-02 rerank cue**: an event string ending in `*` (e.g. `"a player shooting*"`) triggers BLIP ITM reranking — keep that suffix convention when wiring new features.
- **Dual-stream search** (MemoriEase 3.0): when `USE_DUAL_EMBEDDING=true` and `data/blip2_embeddings/` is populated, `EmbeddingIndex` loads a second `blip2_embeddings` array and `SearchEngine._dual_topk_indices` blends `EVA02_SEARCH_WEIGHT * sim_eva02 + BLIP2_SEARCH_WEIGHT * sim_blip2` over a `top_k * DUAL_SEARCH_POOL_MULTIPLIER` candidate pool. Two `IndexFlatIP` indexes are built by `IndexBuilder.build_dual` and pickled together. BLIP-2 text queries go through Q-Former with a 224×224 black image (its `get_qformer_features` is image-only).
- **BLIP-2 VRAM**: ~5 GB on RTX 2080 Ti. Lazy-loaded — `ModelRegistry._load_blip2()` runs on first encode, not at startup. Pre-warm it before a hot search session.
- **Metadata extracted via stub callables by default**: `scripts/smoke_test_metadata.py` runs without a GPU using stub detector/scene/scene-graph — pass `--real-models` to swap in production. Real-models mode skips Qwen unless `bitsandbytes` is installed.
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
- `PLACES365_WEIGHTS_PATH` (directory or `.pth.tar`; default `./models/places365/`)
- `SCENE_GRAPH_MAX_TRIPLES`, `SCENE_GRAPH_MAX_NEW_TOKENS`, `SCENE_GRAPH_QUANTIZATION` (default `4bit`)
- `TRAKE_DEFAULT_TOP_K=200`, `TRAKE_DEFAULT_CANDIDATES=200`, `GAP_KEYFRAME=15`

`aic26 config --show` prints the resolved values.

## Documentation

`mkdocs.yml` configures mkdocs-material with `mkdocstrings` autodoc. Source pages are referenced via nav; build with `pip install -e ".[docs]" && mkdocs serve`.

Design + planning docs in `docs/`:
- `docs/specs/metadata-extraction.md` — original spec, schema AD-3, deviations from lifeXplore / SnapSeek.
- `docs/plans/metadata-extraction.md` — 12-task plan (all completed on `feature/create-metadata`).
- `docs/plans/metadata-improvement-plan.md` — Phase 1–4 roadmap fixing 3 P0 bugs (detection data not aggregated into sidecars, orchestrator uses dummy callables, low coverage). Status: Phase 1 P0s partially done; Qwen stub remains outstanding.