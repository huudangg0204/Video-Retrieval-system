# Spec: Keyframe Metadata Extraction (Scene Graph + Object BBox + Places365)

**Status**: Draft v1
**Date**: 2026-06-26
**Owner**: AIC-26 team
**Related papers**: `docs/paper-knowledge.md` §5 (SnapSeek 3.0) + §2 (lifeXplore)

---

## Objective

Trích xuất metadata phong phú cho mỗi keyframe trong `data/keyframes/`, kết hợp:
- **Hướng SnapSeek 3.0** (Paper #5): scene graph 3-tuple `(subject, predicate, object)` qua MLLM.
- **Hướng lifeXplore** (Paper #2): bounding box position (mở rộng YOLOv8x output) + semantic concept (Places365 scene recognition).

**Mục tiêu cụ thể**:

1. Cho phép query filter theo **vị trí object trong ảnh** (`-o person|position:top-left`).
2. Cho phép query filter theo **scene category** (`-c "kitchen"`).
3. Cho phép query filter theo **quan hệ giữa objects** (`-r "person holding cup"`).
4. Không thay đổi EVA-02 embedding pipeline hiện tại — chỉ thêm metadata sidecar.
5. MVP trên 1-2 video đầu tiên để verify pipeline, chưa scale toàn bộ.

**User stories**:

- *Là người dùng tìm kiếm*, tôi muốn gõ `"a person on the left holding a cup"` và hệ thống lọc ra frame có `person` ở nửa trái ảnh và `person` trong quan hệ `holding` với `cup`.
- *Là người dùng tìm kiếm*, tôi muốn gõ `"in a kitchen"` và hệ thống lọc scene Places365 = `kitchen`.
- *Là dev*, tôi muốn load metadata của 1 keyframe trong < 5ms qua `KeyframeMetadata.load(video, frame_idx)`.

---

## Tech Stack

| Layer | Choice | Lý do |
|---|---|---|
| **MLLM (scene graph)** | `Qwen2.5-VL-7B-Instruct` quantized 4-bit (BitsAndBytes) | RTX 2080 Ti 11GB không chạy nổi FP16; 4-bit ≈ 5GB. Theo SnapSeek §3.5 dùng Qwen2.5-VL-7B. |
| **Object detector (bbox)** | `YOLOv8x-OIV7` (đã có) | Mở rộng thêm `position` field. Đã có trong `pipeline/detection.py`. |
| **Scene recognition** | `EfficientNet-B2` pretrained Places365 (từ lifeXplore §3 Table 1) | Nhẹ (~9MB), chạy CPU được, ~365 scene classes. Paper lifeXplore dùng đúng model này. |
| **Storage** | JSON-per-keyframe | Mirror `data/detections/{video_name}/{frame_idx}.json` pattern. |
| **Pipeline integration** | Click subcommand `aic26 pipeline extract-metadata` | Đặt cạnh `extract-keyframes`, `extract-embeddings`. |
| **Config** | Mở rộng `config/constants.py` + `Settings` (Pydantic) | Theo CLAUDE.md "constants are `Final`, add new knobs to Settings instead". |

---

## Commands

### Mới thêm

```bash
# Run full metadata extraction pipeline (object bbox position + Places365 + scene graph)
aic26 pipeline extract-metadata \
    --input data/keyframes \
    --output data/metadata \
    --device cuda \
    --batch-size 8 \
    --skip-scene-graph     # optional, chỉ chạy bbox + Places365

# Verify metadata cho 1 keyframe cụ thể
aic26 pipeline verify-metadata \
    --video L01_V001 \
    --frame-idx 42

# Inspect stats (số keyframes, scenes, scene graph density)
aic26 pipeline metadata-stats --input data/metadata
```

### Verification

```bash
# Unit tests
pytest tests/unit/test_metadata.py -v
pytest tests/unit/test_scene_graph.py -v
pytest tests/unit/test_places365.py -v

# Integration test (chạy trên 10 keyframes mẫu)
pytest tests/integration/test_metadata_pipeline.py -m "not slow" -v

# Manual smoke test
python scripts/smoke_test_metadata.py --video L01_V001 --limit 50
```

### Lint/format

```bash
ruff check --fix src/aic26/pipeline/metadata.py
ruff format src/aic26/pipeline/metadata.py
mypy src/aic26/pipeline/metadata.py
```

---

## Project Structure

```
src/aic26/pipeline/
├── detection.py          # MỚI: thêm position field (top-left, top-right, ...)
├── places365.py          # MỚI: EfficientNet-B2 inference wrapper
├── scene_graph.py        # MỚI: Qwen2.5-VL-7B prompt + parse 3-tuple
├── metadata.py           # MỚI: orchestrator — combine bbox + places + scene graph → JSON sidecar
├── orchestrator.py       # Mở rộng: thêm step 6 "extract-metadata"
└── __init__.py

src/aic26/core/
└── embeddings.py         # Có thể mở rộng load metadata vào EmbeddingIndex (P2, không trong MVP)

src/aic26/config/
└── constants.py          # MỚI: SCENE_GRAPH_PROMPT, POSITION_GRID_3x3, PLACES365_LABELS_PATH
└── (settings.py via Pydantic)  # MỚI: METADATA_DIR, SCENE_GRAPH_MODEL, PLACES365_MODEL_NAME

src/aic26/cli/main.py     # Mở rộng: pipeline subcommand extract-metadata

tests/
├── unit/
│   ├── test_metadata.py
│   ├── test_scene_graph.py
│   ├── test_places365.py
│   └── test_position.py
├── integration/
│   └── test_metadata_pipeline.py
└── fixtures/
    └── metadata/
        ├── L01_V001_000.json
        └── L01_V001_010.json

data/
└── metadata/             # MỚI: mirror data/keyframes/ structure
    └── L01_V001/
        ├── 000.json
        ├── 010.json
        └── ...

scripts/
└── smoke_test_metadata.py  # MỚI

docs/
├── specs/metadata-extraction.md   # file này
└── paper-knowledge.md             # đã có
```

---

## Code Style

Một ví dụ đại diện cho module con (`scene_graph.py`):

```python
"""Scene graph extraction via Qwen2.5-VL-7B.

Implements the 3-tuple (subject, predicate, object) representation
from SnapSeek 3.0 §3.5 (LSC'25). Output is JSON-serializable.

Why: enable relational queries like "person holding cup" without
relying on CLIP embedding to capture object interactions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

from PIL import Image

logger = logging.getLogger(__name__)

# Lower-case noun, base-form predicates, no passive constructions (SnapSeek §3.5 prompt rules)
SCENE_GRAPH_PROMPT: Final[str] = """Analyze this image and list all visible subject-predicate-object triples.

Rules:
- Use lowercase nouns for subjects and objects (e.g., 'person', 'cup', 'table')
- Use base-form verbs for predicates (e.g., 'hold', 'sit on', 'next to', 'in front of')
- Avoid passive constructions
- Only describe triples you can clearly see
- Limit to the top 5 most salient triples

Output format (JSON only, no commentary):
{"triples": [["subject", "predicate", "object"], ...]}"""

MODEL_NAME: Final[str] = "Qwen/Qwen2.5-VL-7B-Instruct"


def extract_scene_graph(image: Image.Image, model) -> list[tuple[str, str, str]]:
    """Extract (subject, predicate, object) triples from one image.

    Args:
        image: PIL Image, RGB, any size.
        model: Loaded Qwen2.5-VL-7B pipeline.

    Returns:
        Up to 5 (subject, predicate, object) tuples, lowercase.

    Raises:
        ValueError: If model output is not valid JSON.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": SCENE_GRAPH_PROMPT},
            ],
        }
    ]
    response = model.chat(messages, max_new_tokens=512)

    try:
        data = json.loads(response)
        triples = [tuple(t) for t in data["triples"]]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Bad scene graph output: %s", response)
        raise ValueError(f"Invalid scene graph JSON: {exc}") from exc

    return triples[:5]


def save_scene_graph(path: Path, triples: list[tuple[str, str, str]]) -> None:
    """Persist triples to JSON sidecar matching the detection format."""
    payload = {"scene_graph": [{"s": s, "p": p, "o": o} for s, p, o in triples]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
```

**Key conventions**:

- Module-level `Final[str]` for prompts & magic strings.
- `from __future__ import annotations` cho clean type hints.
- Triple được normalize thành lowercase trước khi lưu.
- Module docstring cite paper section.
- Functions nhỏ, single-purpose, dễ test.
- `logger.warning` thay vì raise trên malformed model output (recoverable).

---

## Testing Strategy

| Layer | Framework | Location | Coverage |
|---|---|---|---|
| Unit | pytest | `tests/unit/` | Mỗi function riêng, mock model output. |
| Integration | pytest | `tests/integration/` | End-to-end trên 10 keyframes mẫu, GPU optional (mark `@pytest.mark.gpu`). |
| Smoke | script | `scripts/smoke_test_metadata.py` | Manual run trên 1 video đầu. |

**Markers**: dùng `gpu` cho tests cần GPU; `slow` cho tests > 5s.

**Fixtures**:

```python
# tests/conftest.py
@pytest.fixture
def sample_keyframe(tmp_path) -> Path:
    """Create a 640x360 dummy keyframe with 2 visible objects."""
    ...

@pytest.fixture
def mock_qwen_output() -> str:
    return '{"triples": [["person", "hold", "cup"], ["person", "sit on", "chair"]]}'

@pytest.fixture
def mock_places365_labels() -> list[str]:
    return ["kitchen", "bedroom", "outdoor", ...]
```

**Coverage target**: ≥ 80% cho `pipeline/metadata.py`, ≥ 90% cho `pipeline/places365.py`.

**Key test cases**:

1. `extract_scene_graph` parse JSON đúng → trả về list triples.
2. `extract_scene_graph` model output malformed → log warning + return `[]`.
3. `Places365Classifier` classify ảnh test → trả về top-5 categories hợp lý.
4. `position_from_bbox([10, 10, 100, 200])` ở ảnh 640x360 → `bottom-left` (vì x_center < 0.33).
5. End-to-end: load keyframe → run pipeline → JSON sidecar chứa cả 3 fields.
6. JSON sidecar backward-compatible: nếu scene_graph skip thì field đó = `null`.

---

## Boundaries

### Always do

- Luôn chạy `pytest tests/unit/test_metadata*.py -v` trước commit.
- Luôn validate JSON output của Qwen trước khi lưu (không crash pipeline nếu 1 ảnh fail).
- Log progress mỗi 100 keyframes (`logger.info`).
- Cite paper section trong docstring của function mới.
- Add new env var vào `.env.example` nếu dùng setting.
- Update `docs/paper-knowledge.md` nếu thêm feature mới từ paper.

### Ask first

- Thêm dependency mới (transformers >= 4.45 cho Qwen2.5-VL, bitsandbytes, accelerate).
- Thay đổi `pipeline/detection.py` (existing module) — chỉ thêm field, không sửa logic cũ.
- Tăng default batch size (ảnh hưởng VRAM).
- Thay đổi format JSON sidecar (breaking change cho downstream code).

### Never do

- Modify EVA-02 embedding pipeline (`core/embeddings.py`, `pipeline/embedding.py`).
- Re-train Qwen hoặc Places365 — chỉ inference.
- Save model weights vào repo (chỉ tên + cache).
- Skip validation khi gặp bad output — luôn handle gracefully.
- Auto-commit khi CI đang fail.

---

## Success Criteria

### MVP (1-2 ngày)

- [ ] `aic26 pipeline extract-metadata --input data/keyframes --output data/metadata` chạy thành công trên 1 video (≥ 50 keyframes).
- [ ] Mỗi keyframe có JSON sidecar chứa 3 fields: `detections`, `scene`, `scene_graph`.
- [ ] `pytest tests/unit/` pass ≥ 80% coverage cho module mới.
- [ ] `verify-metadata --video L01_V001 --frame-idx 42` in ra JSON đẹp, dễ đọc.
- [ ] `metadata-stats` in ra: tổng keyframes, scene category distribution, average scene graph triples/keyframe.
- [ ] Smoke test script chạy được trên 50 keyframes đầu của `L01_V001` trong < 5 phút (GPU RTX 2080 Ti).
- [ ] Zero crash khi gặp ảnh corrupt (skip + log warning).

### Out of scope cho MVP

- ❌ Integration với retrieval pipeline (`core/embeddings.py` đọc metadata để filter).
- ❌ Scale toàn bộ `data/keyframes/` (chỉ chạy 1-2 video).
- ❌ ADL classification (35 classes) — quyết định chỉ scene graph.
- ❌ Roaring Bitmaps filter (đề xuất từ Libro).
- ❌ UI mới trong Flask GUI để filter theo scene/position/relation.

### Stretch (sau MVP)

- [ ] Add filter `-r "person holding cup"` vào `core/search.py::_filter_by_scene_graph`.
- [ ] Add filter `-c kitchen` vào search.
- [ ] Add position filter `-o person|position:top-left`.
- [ ] Integration test với end-to-end retrieval: query có filter → recall top-10 có ≥ 8 relevant.

---

## Open Questions

1. **Qwen2.5-VL-7B quantization**: 4-bit (bitsandbytes) hay AWQ? 4-bit đơn giản hơn, AWQ nhanh hơn nhưng cần thêm setup.
   → *Default*: bitsandbytes 4-bit. Tune sau nếu quá chậm.

2. **Scene graph cache**: nếu chạy lại pipeline, có cần skip frames đã có scene_graph?
   → *Default*: skip nếu JSON đã có field `scene_graph` (idempotent re-run).

3. **Bounding box position grid**: 3×3 (top-left, top, top-right, left, center, right, bottom-left, bottom, bottom-right) hay chỉ 4 góc?
   → *Default*: 3×3 cho granular hơn.

4. **Places365 model hosting**: download về `models/places365/` (gitignored) hay dùng `torchvision` pretrained?
   → *Default*: download 1 lần qua `aic26 download places365`, cache vào `models/places365/`.

5. **Output format JSON scene_graph**: array of objects `[{s,p,o}, ...]` hay array of arrays `[[s,p,o], ...]`?
   → *Default*: object form dễ đọc hơn, dễ extend thêm field sau.

---

## Verification Checklist (trước khi merge)

- [ ] `pre-commit run --all-files` pass.
- [ ] `pytest -m "not slow and not gpu"` pass.
- [ ] `pytest --cov=aic26.pipeline.metadata --cov-report=term-missing` ≥ 80%.
- [ ] `aic26 pipeline metadata-stats` chạy được trên output thật.
- [ ] `docs/paper-knowledge.md` có link đến spec này ở mục SnapSeek / lifeXplore.
- [ ] README.md có dòng về `extract-metadata` subcommand (optional, có thể bỏ qua MVP).
