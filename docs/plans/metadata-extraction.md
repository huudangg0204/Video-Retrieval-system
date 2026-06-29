# Plan: Keyframe Metadata Extraction

**Status**: Draft v1 (chờ review)
**Date**: 2026-06-26
**Parent spec**: [`docs/specs/metadata-extraction.md`](../specs/metadata-extraction.md)

---

## Architectural Decisions

### AD-1: MLLM runtime chọn transformers + bitsandbytes, không dùng vLLM

**Context**: Qwen2.5-VL-7B cần ~14GB FP16, RTX 2080 Ti chỉ có 11GB.

**Options considered**:
- A. `transformers` + `bitsandbytes` 4-bit (~5GB) ← **chọn**
- B. `vLLM` — nhanh hơn 10-20× nhưng cần GPU ≥ 24GB (không khả thi)
- C. `ollama` — đơn giản nhưng thêm dependency runtime, khó debug

**Decision**: A. Trade-off: chậm hơn vLLM (~2-5s/keyframe thay vì 0.3s) nhưng chạy được trên hardware hiện tại.

### AD-2: Places365 inference wrapper độc lập, không qua `pipeline/detection.py`

**Context**: YOLOv8x OIV7 detect object bbox; Places365 classify scene category. Hai tác vụ khác nhau.

**Options considered**:
- A. Module riêng `pipeline/places365.py` ← **chọn**
- B. Mở rộng `pipeline/detection.py` thêm scene classification

**Decision**: A. Tách rời để dễ test, dễ swap model sau này (vd: chuyển sang CLIP zero-shot scene). `detection.py` chỉ thêm field `position` (low-risk extension).

### AD-3: Scene graph JSON key = `scene_graph`, structure = array of objects

**Context**: paper SnapSeek dùng `<sub><pred><obj>` text; AIC-26 cần structured.

**Schema**:
```json
{
  "detections": [{"label": "person", "score": 0.92, "bbox": [10, 20, 100, 200], "position": "top-left"}],
  "scene": {"category": "kitchen", "confidence": 0.85, "top5": [...]},
  "scene_graph": [{"s": "person", "p": "hold", "o": "cup", "confidence": 0.9}]
}
```

**Decision**: 3 keys top-level. Backward-compatible (nếu field = null, downstream xử lý OK).

### AD-4: CLI integration qua Click group, không phải standalone script

**Context**: CLAUDE.md nói "Single entry point" và hiện có pattern `aic26 pipeline <subcommand>`.

**Decision**: Mở rộng `src/aic26/cli/main.py` thêm 3 subcommands:
- `aic26 pipeline extract-metadata`
- `aic26 pipeline verify-metadata`
- `aic26 pipeline metadata-stats`

### AD-5: Orchestrator không tự động chạy metadata trong `run-all`

**Context**: Spec MVP chỉ chạy 1-2 video; auto-include trong `run-all` sẽ chậm toàn pipeline.

**Decision**: `run-all` thêm flag `--extract-metadata` (default: OFF). User phải opt-in.

---

## Dependency Graph

```
[Task 1: Constants + Settings]
        │
        ▼
[Task 2: Position helper] ◀── [Task 3: Places365 module]
        │                              │
        ▼                              │
[Task 4: Extend detection.py]          │
        │                              │
        └──────────┬───────────────────┘
                   ▼
        [Task 5: Scene graph module]
                   │
                   ▼
        [Task 6: Metadata orchestrator]
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    [Task 7]  [Task 8]   [Task 9]
    CLI       Tests      Smoke
```

**Parallelism**: Task 2 (position) và Task 3 (Places365) có thể chạy song song vì không phụ thuộc nhau. Task 5 (scene graph) chỉ cần Task 1 (constants). Test (Task 8) phải đợi Task 5-6 xong.

---

## Implementation Order

### Phase A: Foundation (ngày 1 — sáng)

| # | Task | File(s) | Output |
|---|---|---|---|
| 1 | Add constants + Settings | `config/constants.py`, `config/settings.py` | `POSITION_GRID_3x3`, `SCENE_GRAPH_PROMPT`, `PLACES365_MODEL_NAME`, `METADATA_DIR` |
| 2 | Position helper | `pipeline/position.py` (mới) | `position_from_bbox(bbox, w, h) → str` |
| 3 | Places365 module | `pipeline/places365.py` (mới) | `Places365Classifier` class với `classify(image) → top5` |

### Phase B: Scene graph (ngày 1 — chiều)

| # | Task | File(s) | Output |
|---|---|---|---|
| 4 | Extend `detection.py` | `pipeline/detection.py` | Thêm `position` field vào output dict |
| 5 | Scene graph module | `pipeline/scene_graph.py` (mới) | `extract_scene_graph(image, model) → list[tuple]` |
| 6 | Metadata orchestrator | `pipeline/metadata.py` (mới) | `MetadataExtractor.run(input, output, ...)` |

### Phase C: CLI + Tests (ngày 2 — sáng)

| # | Task | File(s) | Output |
|---|---|---|---|
| 7 | CLI subcommands | `cli/main.py` | 3 subcommands mới |
| 8 | Unit tests | `tests/unit/test_*.py` | ≥ 80% coverage |
| 9 | Smoke test script | `scripts/smoke_test_metadata.py` | End-to-end trên 50 keyframes |

### Phase D: MVP validation (ngày 2 — chiều)

| # | Task | File(s) | Output |
|---|---|---|---|
| 10 | Download Places365 weights | `scripts/download_places365.py` (mới) | `models/places365/*.pth` |
| 11 | Run trên `L01_V001` thật | manual | `data/metadata/L01_V001/*.json` |
| 12 | Verify output | `metadata-stats` + manual JSON inspect | Stats + sanity check |

---

## Risks & Mitigations

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | Qwen2.5-VL-7B không load được trên RTX 2080 Ti 11GB dù 4-bit | High — blocker | Fallback sang **SmolVLM-Instruct** (256M params, chạy FP16 thoải mái) hoặc **BLIP-2 ITM** (đã có sẵn). Detect tại runtime: nếu OOM → log warning + skip scene_graph + để fields = null. |
| R2 | Places365 weights download chậm hoặc fail mirror | Medium — blocker download | Use official MIT-CSAIL URL `http://places.csail.mit.edu/...`. Nếu fail: skip scene classification, return `null`. |
| R3 | Qwen output không phải JSON hợp lệ (model hallucinate) | Medium — silent corrupt | Robust JSON parser: regex extract JSON block, retry 1 lần với `temperature=0`. Log bad outputs. |
| R4 | Memory leak khi batch process 1000+ keyframes với Qwen | Medium — OOM after N frames | Batch size = 1 cho Qwen (giữ model ở GPU giữa các frames), explicit `torch.cuda.empty_cache()` mỗi 50 frames. |
| R5 | Bbox position ambiguous khi object rất nhỏ | Low — bad UX | Skip position cho bbox width/height < 5% image size. Return `null`. |
| R6 | JSON sidecar size lớn (scene_graph có thể 10 triples × fields = ~500 bytes × 50K = 25MB) | Low — disk | Acceptable. Nếu cần compact: dùng msgpack hoặc flatten thành CSV. |

---

## Parallel Work

Có thể làm song song (nếu nhiều người):
- **Track A**: Task 2 (position) + Task 4 (extend detection) — bbox-related.
- **Track B**: Task 3 (Places365) — independent.
- **Track C**: Task 5 (scene graph) + download model weights — MLLM-related.

Track A và B không phụ thuộc nhau. Track C phụ thuộc Task 1.

---

## Verification Checkpoints

Sau mỗi phase, verify rồi mới qua phase tiếp:

### Checkpoint A (sau Task 1-3)
- [ ] `from aic26.pipeline.position import position_from_bbox` import OK.
- [ ] `from aic26.pipeline.places365 import Places365Classifier` import OK.
- [ ] Unit test `test_position.py` pass (3×3 grid test cases).
- [ ] Unit test `test_places365.py` pass với dummy weights.

### Checkpoint B (sau Task 4-6)
- [ ] `metadata.py` `run()` function callable end-to-end trên 1 ảnh test.
- [ ] Output JSON chứa đủ 3 fields: `detections`, `scene`, `scene_graph`.
- [ ] Idempotent: chạy 2 lần trên cùng input → output giống hệt.

### Checkpoint C (sau Task 7-9)
- [ ] `aic26 pipeline extract-metadata --help` hiển thị đúng flags.
- [ ] `pytest tests/unit/test_metadata*.py -v` pass, coverage ≥ 80%.
- [ ] `python scripts/smoke_test_metadata.py --video L01_V001 --limit 50` chạy trong < 5 phút.

### Checkpoint D (sau Task 10-12) — MVP DONE
- [ ] `data/metadata/L01_V001/` có ≥ 50 JSON files.
- [ ] `metadata-stats` in ra scene distribution và scene_graph density.
- [ ] 3 file JSON đầu tiên được manual inspect: format đúng, không có null không mong muốn.

---

## Out of Plan (ghi nhớ cho sau MVP)

Đã confirm với user là **không** nằm trong MVP:

1. ❌ Integration với `core/embeddings.py` (filter search theo metadata).
2. ❌ CLI flag mới `-r`, `-c`, `-o|position` trong `search` subcommand.
3. ❌ Roaring Bitmaps index (đề xuất từ Libro).
4. ❌ ADL classification (35 classes — bị user loại bỏ ở câu hỏi 1).
5. ❌ GUI mới trong Flask để filter theo scene/position/relation.

Có thể track ở GitHub Issues hoặc backlog comment trong `paper-knowledge.md`.
