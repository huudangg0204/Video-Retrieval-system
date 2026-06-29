# 📋 Kế hoạch Cải thiện Metadata — K05_V004

> **Ngày tạo:** 2026-06-29
> **Video mục tiêu phân tích:** K05_V004
> **Áp dụng cho:** Toàn bộ pipeline metadata AIC-26
> **Trạng thái tổng quan:**
> - Phase 1 (P0 detections): ✅ HOÀN THÀNH (2026-06-29) — committed `e4b409f`
> - Phase 2 (P1 scene improvements): ✅ HOÀN THÀNH (2026-06-29) — committed `e4b409f`
> - Phase 3.1 (OCR): ✅ HOÀN THÀNH (2026-06-29) — committed `e4b409f`
> - Phase 3.2 (BLIP captioning): ✅ HOÀN THÀNH (2026-06-29) — on disk + tests, chưa commit
> - Phase 3.3 (Temporal context): ⏳ CHƯA BẮT ĐẦU
> - Phase 4 (Optimizations): ⏳ CHƯA BẮT ĐẦU

---

## 1. Tổng quan Phân tích Hiện trạng (updated 2026-06-29)

### 1.1 Dữ liệu Keyframes — K05_V004 (sau khi hoàn thành Phase 1, 2, 3.1, 3.2)

| Chỉ số | Giá trị |
|---|---|
| Tổng keyframes | **427 frames** |
| Metadata sidecars đã tạo | **427 / 427** (100%) |
| Detection files đã tạo | **426 / 427** (99.8%) |
| Sidecars có `detections` w/ `position` | **377 / 427** (88.3%) |
| Sidecars có `scene` (Places365) | **427 / 427** (100%) |
| Sidecars có `scene_group` | **427 / 427** (100%) |
| Sidecars có `smoothed_scene` | **427 / 427** (100%) |
| Sidecars có `scene_graph` (Qwen) | **0 / 427** (0%) ⚠️ |
| Sidecars có `ocr_text` | **427 / 427** (100%) |
| Sidecars có `caption` (BLIP) | **427 / 427** (100%) |

### 1.2 Phân bố Scene sau khi áp dụng Phase 2

| Metric | Trước Phase 2 | Sau Phase 2 |
|---|---|---|
| Unique Places365 leaf categories | 82 | 36 (sau smoothing) |
| Unique `scene_group` | n/a | 13 |
| Top `scene_group` | n/a | `outdoor_urban` (79), `indoor_public` (68), `other` (66), `entertainment_culture` (65) |
| Top `smoothed_scene` | n/a | `uncertain` (289), `television_studio` (20), `airport_terminal` (11), `booth_indoor` (11), `auto_showroom` (10) |

**Quan sát quan trọng:** Số 28 categories "đáng ngờ" trong phân tích ban đầu thực ra là do **thiếu temporal smoothing**. Sau khi áp dụng majority-vote sliding window (Phase 2.3), số unique scene giảm từ 82 → 36 và top result hợp lý (`uncertain` chiếm 289 frames vì đây là broadcast video nhiều cảnh trộn lẫn). Sự "phân tán" của Places365 không phải bug model, mà là do TV-style cuts trong video.

### 1.3 OCR coverage — Phase 3.1

- **3,139 text regions** đã được merge vào sidecar (avg ~7.35 regions/frame).
- Chủ yếu là chú thích tiếng Việt + timestamp overlay (vd: `"(NL) Kết quả bốc thăm AFF Cup 2024"`, `"CHUYỂN ĐỘNG 24H"`).
- Schema: `ocr_text: [{"text": str, "bbox": [x1,y1,x2,y2], "confidence": float}]` (xyxy sau khi convert từ polygon 4 điểm của EasyOCR).

### 1.4 Caption coverage — Phase 3.2

- **427/427 sidecars** có `caption` (string, English).
- Trung bình **46 chars / caption**, 0 captions trống.
- Model: `Salesforce/blip-image-captioning-base` (446M params, ~2GB VRAM, ~0.3s/frame trên RTX 2080 Ti).
- Sample spot-checks: tất cả đều là câu tiếng Anh mạch lạc, mô tả nội dung frame.

---

## 2. Vấn đề và Kế hoạch

### 2.1 Các vấn đề P0 ban đầu

#### 🔴 P0 — Metadata Coverage cực thấp (11.7%) — ✅ ĐÃ GIẢI QUYẾT

**Trước:** 50/427 sidecars (11.7%).
**Sau Phase 1.3:** 427/427 (100%).
**Cách làm:**
- `MetadataExtractor` giờ đọc pre-computed detections từ `data/detections/{video}/{frame}.json` thay vì chạy lại YOLO (`load_precomputed_detections()` trong `pipeline/metadata.py`).
- CLI command `aic26 pipeline extract-metadata --input data/keyframes --output data/metadata` (chạy độc lập, không cần `run-all`).

#### 🔴 P0 — Detections hoàn toàn trống trong Metadata (100%) — ✅ ĐÃ GIẢI QUYẾT

**Trước:** 50/50 sidecars có `detections: []`.
**Sau Phase 1:** 377/427 (88.3%) sidecars có detections với `position` field. 50 còn lại là frames YOLO thực sự không detect được object nào.
**Cách làm:**
- `KeyframeMetadata.detections` schema giờ gồm `position` (3×3 grid từ `pipeline/position.py`).
- Bbox < 5% diện tích ảnh trả về `position: None` (theo spec R5).
- 3×3 grid: `top-left | top-center | top-right | center-left | center | center-right | bottom-left | bottom-center | bottom-right`.

#### 🔴 P0 — Scene Graph hoàn toàn trống (100%) — ⚠️ VẪN CÒN (0%)

**Hiện trạng:** Tất cả 427 sidecars có `scene_graph: []`. Qwen2.5-VL model **chưa bao giờ được invoke end-to-end** trên K05_V004 vì:
- `PipelineOrchestrator.run(extract_metadata=True)` vẫn pass dummy `scene_graph_fn=lambda img: []` (orchestrator chưa wire real callable).
- Code `pipeline/scene_graph.py` đã sẵn sàng (extract_scene_graph + Qwen2.5-VL-7B-Instruct 4-bit qua bitsandbytes), nhưng chưa được execute vì cần GPU ~14GB VRAM.

**Tác động:** Truy vấn relational kiểu "person hold cup", "car park on street" hoàn toàn bất khả thi.
**Bước tiếp theo:** Phase 4.1 — chạy batch Qwen2.5-VL inference thực tế trên K05_V004.

### 2.2 Các vấn đề P1

#### 🟡 P1 — Scene Classification confidence thấp (70%) — ✅ ĐÃ GIẢI QUYẾT một phần

**Trước:** 35/50 frames (70%) có scene confidence < 0.5.
**Sau Phase 2.2:** Khi top-1 confidence < 0.3, scene được gán là `"uncertain"` → có fallback hợp lý.
**Sau Phase 2.3:** Temporal smoothing với sliding window = 5 majority vote → unique scene giảm 82 → 36.

#### 🟡 P1 — Top-5 Scenes có duplicate category — ✅ ĐÃ GIẢI QUYẾT

**Cách làm (Phase 2.1):** De-duplicate leaf labels bằng cách append parent path khi có collision. Ví dụ:
- `/p/platform/train_station_platform` → `train_station_platform`
- `/p/platform/outdoor` → `platform` (giữ nguyên vì chưa collision)

Logic nằm trong `Places365Classifier._infer_scene_group()` và label-loading code.

### 2.3 Các vấn đề P2

#### 🟢 P2 — Thiếu các modalities bổ sung — 🔄 ĐANG XỬ LÝ

| Modality | Status | Ghi chú |
|---|---|---|
| OCR text | ✅ Phase 3.1 done | 3,139 regions merged end-to-end |
| Image captions (English) | ✅ Phase 3.2 done | 427/427 captions via BLIP base |
| Image captions (Vietnamese) | ⏳ Deferred (3.2.2) | Cần VinAI/visobert-ViCaptioning |
| Caption-based text search | ⏳ Deferred (3.2.1) | Touches core/search.py — integration pending |
| Audio/ASR transcription | ⏳ Out of scope | |
| Action recognition | ⏳ Out of scope | |
| Temporal context | ⏳ Phase 3.3 next | |

---

## 3. Kế hoạch chi tiết — từng Phase

### Phase 1: Khắc phục Khẩn cấp (P0) — ✅ HOÀN THÀNH

#### Task 1.1: Tích hợp Detection Data vào Metadata Sidecar — ✅ DONE
**File:** `src/aic26/pipeline/metadata.py`
- `load_precomputed_detections(detection_dir) -> DetectorFn`: factory tạo callable đọc `data/detections/{video}/{frame}.json`.
- `MetadataExtractor.__init__` nhận `detector_fn` (default = `load_precomputed_detections(settings.detection_dir)`).
- `KeyframeMetadata.detections: list[Detection]` giờ chứa `position` field.

#### Task 1.2: Wire Real Models vào Orchestrator — 🔄 PARTIAL
**File:** `src/aic26/pipeline/orchestrator.py`
- `PipelineOrchestrator.run(extract_metadata=False)` mới có flag opt-in (AD-5).
- Hiện pass detector_fn (đọc từ data/detections/) ✅
- Hiện pass scene_fn thực (Places365Classifier) ✅
- **Còn lại:** scene_graph_fn vẫn dummy `lambda img: []` ⚠️ (Phase 4.1 sẽ wire Qwen)

#### Task 1.3: Chạy lại Metadata Extraction — ✅ DONE
- Command: `aic26 pipeline extract-metadata -i data/keyframes -o data/metadata [--overwrite]`
- 427/427 sidecars đã tạo trên K05_V004.

---

### Phase 2: Nâng cao Scene Classification (P1) — ✅ HOÀN THÀNH

#### Task 2.1: De-duplicate Places365 Labels — ✅ DONE
**File:** `src/aic26/pipeline/places365.py`
- Trong `Places365Classifier._infer_scene_group(label)`: append parent path khi collision.
- Top categories không còn duplicate.

#### Task 2.2: Scene Confidence Thresholding + Fallback — ✅ DONE
- Khi top-1 confidence < 0.3 → scene.category = `"uncertain"`.
- `scene_group` aggregate các categories vào 13 nhóm: `outdoor_urban`, `indoor_public`, `entertainment_culture`, `transportation`, `sports_recreation`, ...

#### Task 2.3: Scene Temporal Smoothing — ✅ DONE
**File:** `src/aic26/pipeline/metadata.py::apply_temporal_smoothing()`
- Sliding window = 5, majority vote.
- `smoothed_scene` field trong sidecar.
- Kết quả: 82 unique → 36 unique categories, top: `uncertain` (289), `television_studio` (20), `airport_terminal` (11).

---

### Phase 3: Bổ sung Modalities (P2) — 🔄 MIXED

#### Task 3.1: Tích hợp OCR vào Metadata — ✅ HOÀN THÀNH
**Files:** `src/aic26/pipeline/ocr.py`, `src/aic26/pipeline/metadata.py`, `src/aic26/cli/main.py`
- Polygon bbox `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]` (EasyOCR output) → xyxy `[x_min, y_min, x_max, y_max]`.
- Helper `_polygon_to_xyxy()` trong `pipeline/ocr.py`.
- CLI command: `aic26 pipeline merge-ocr --input data/ocr_results.csv --metadata-dir data/metadata [--video V] [--dry-run]`.
- `KeyframeMetadata.ocr_text: list[OCRText] | None` — None khi chưa merge.
- Schema: `{"text": str, "bbox": [x1,y1,x2,y2], "confidence": float}`.
- Tests: `test_ocr_polygon.py` (4 tests), tests trong `test_cli_metadata.py` (4 tests cho merge-ocr).
- End-to-end: 427/427 sidecars updated, 3,139 text regions merged.

#### Task 3.2: Image Captioning — ✅ HOÀN THÀNH
**Files mới:**
- `src/aic26/pipeline/captioning.py` — `CaptionGenerator` + `build_caption_fn()`.
- `tests/unit/test_captioning.py` — 12 tests (mock BLIP/BLIP-2 classes ở boundary `transformers.*`).

**Files sửa:**
- `src/aic26/pipeline/metadata.py` — `CaptionFn` type alias, `KeyframeMetadata.caption`, `MetadataStats.sidecars_with_caption`, `MetadataStats.caption_total_chars`, `build_metadata_callables()` returns 4-tuple, `apply_caption_merging()` post-pass function.
- `src/aic26/cli/main.py` — `--skip-caption` flag, `merge-captions` subcommand.
- `src/aic26/config/settings.py` — `caption_model_name`, `caption_max_new_tokens`, `caption_min_length`, `caption_quantization`.
- `src/aic26/config/constants.py` — `CAPTION_DEFAULT_MODEL`, `CAPTION_PROMPT`, `CAPTION_MAX_NEW_TOKENS`, `CAPTION_MIN_LENGTH`.

**Model đã chọn:** `Salesforce/blip-image-captioning-base` (446M params) — **KHÔNG dùng** `Salesforce/blip2-itm-vit-g` (ITM variant — không có `.generate()` method, đã được document trong memory `blip2-itm-g-image-only`).

**Implementation:**
- `CaptionGenerator` lazy-loads model+processor trên first `.generate()` call.
- Branches trên `"blip2" in model_name.lower()` để chọn `BlipProcessor`/`BlipForConditionalGeneration` (BLIP base) hoặc `Blip2Processor`/`Blip2ForConditionalGeneration` (BLIP-2).
- Returns `""` on any exception (graceful degradation).
- Default prompt: `"a photo of"` (BLIP base ignores prompt nhưng wiring vẫn truyền qua processor).

**CLI:**
- `aic26 pipeline extract-metadata --input data/keyframes --output data/metadata --skip-caption` (skip flag mới).
- `aic26 pipeline merge-captions --input data/metadata --keyframes-dir data/keyframes [--video V] [--model-name NAME] [--dry-run]`.

**End-to-end:** 427/427 K05_V004 sidecars updated, avg 46 chars/caption, 0 empty captions.

#### Task 3.3: Temporal Context — ⏳ CHƯA BẮT ĐẦU
```
Hành động:
- Thêm fields:
    "prev_frame_idx": int | null,
    "next_frame_idx": int | null,
    "temporal_segment_id": int  # nhóm frames liên tục cùng scene
- Cho phép navigation giữa các frames liên quan
- Lợi dụng: `smoothed_scene` runs (consecutive frames cùng smoothed_scene) tạo thành natural segments
```

---

### Phase 4: Tối ưu Pipeline (P2) — ⏳ CHƯA BẮT ĐẦU

#### Task 4.1: Batch Scene Graph Inference — ⏳ NEXT
```
Hành động:
- Chạy Qwen2.5-VL-7B-Instruct (4-bit qua bitsandbytes) trên toàn bộ K05_V004
- Code: pipeline/scene_graph.py đã sẵn sàng, cần wire vào orchestrator
- Ước tính: 427 frames × 3s = ~21 phút (single-batch) → batch 4 = ~5-7 phút
- Output: scene_graph: [{s, p, o, confidence}] cho 427 sidecars
```

#### Task 4.2: Incremental Metadata Update — ⏳ PENDING
```
Hành động:
- Hiện process_video() skip file đã tồn tại khi overwrite=False
- Thêm mode "merge": đọc sidecar hiện có, chỉ cập nhật fields còn thiếu
  Ví dụ: sidecar có scene nhưng thiếu caption → chỉ chạy caption_fn
- Giảm thời gian khi thêm modality mới (vd: sau khi Qwen wired → chỉ chạy scene_graph pass)
```

#### Task 4.3: Schema Validation + Migration — ⏳ PENDING
```
Hành động:
- Tạo JSON Schema chính thức cho metadata sidecar
- Script migrate cũ → schema mới (backward-compatible)
- CI test validate tất cả sidecars against schema
- Hiện tại backward-compat được thực hiện bằng "only emit when populated" — không có version field
```

---

## 4. Schema Metadata hiện tại (post-Phase 3.2)

**Lưu ý:** KHÔNG có `version` field ở top-level. Backward-compat được thực hiện bằng cách **chỉ emit optional fields khi populated** (None thì không xuất hiện trong JSON). Đây là chiến lược khác với ban đầu đề xuất (`"version": "2.0"`).

```json
{
  "video": "K05_V004",
  "frame_idx": 42,
  "detections": [
    {
      "label": "Person",
      "score": 0.92,
      "bbox": [10, 20, 100, 200],
      "position": "top-left"
    }
  ],
  "scene": {
    "category": "airport_terminal",
    "confidence": 0.633,
    "scene_group": "transportation",
    "top5": [["airport_terminal", 0.633], ["indoor", 0.185]],
    "smoothed_category": "airport_terminal"
  },
  "scene_graph": [
    {"s": "person", "p": "hold", "o": "cup", "confidence": 0.9}
  ],
  "ocr_text": [
    {"text": "EXIT", "bbox": [50, 10, 120, 40], "confidence": 0.95}
  ],
  "caption": "A person holding a cup while standing in an airport terminal."
}
```

**Quy tắc backward-compat:**
- `detections`: luôn emit (mặc định `[]`).
- `scene`: `null` khi chưa chạy, `{...}` khi đã chạy.
- `scene_graph`: `[]` khi chưa chạy hoặc ran-but-empty.
- `ocr_text`: `null` khi chưa merge, `[{...}]` sau merge.
- `caption`: `null` khi chưa chạy, `""` (chuỗi rỗng) khi model thất bại — phân biệt với model chưa chạy.

---

## 5. Ưu tiên Thực hiện — Updated Effort Table

| # | Task | Priority | Effort | Impact | Status |
|---|---|---|---|---|---|
| 1.1 | Tích hợp Detection → Metadata | 🔴 P0 | 3h | Rất cao | ✅ DONE |
| 1.2 | Wire real models vào orchestrator | 🔴 P0 | 4h | Rất cao | 🔄 PARTIAL (Qwen chưa wire) |
| 1.3 | Chạy lại metadata extraction | 🔴 P0 | 1h | Rất cao | ✅ DONE |
| 2.1 | De-duplicate Places365 labels | 🟡 P1 | 2h | Trung bình | ✅ DONE |
| 2.2 | Scene confidence thresholding | 🟡 P1 | 3h | Trung bình | ✅ DONE |
| 2.3 | Scene temporal smoothing | 🟡 P1 | 4h | Cao | ✅ DONE |
| 3.1 | Tích hợp OCR | 🟢 P2 | 3h | Cao | ✅ DONE |
| 3.2 | Image captioning (BLIP base) | 🟢 P2 | 6h | Cao | ✅ DONE (chưa commit) |
| 3.3 | Temporal context | 🟢 P2 | 3h | Trung bình | ⏳ NEXT |
| 4.1 | Batch scene graph (Qwen) | 🟢 P2 | 4h | Trung bình | ⏳ PENDING |
| 4.2 | Incremental update mode | 🟢 P2 | 4h | Cao | ⏳ PENDING |
| 4.3 | Schema validation + migration | 🟢 P2 | 3h | Trung bình | ⏳ PENDING |

**Tổng effort ước tính:** ~40h (~1 tuần full-time).
**Effort đã hoàn thành:** ~26h (Phase 1 + 2 + 3.1 + 3.2).
**Effort còn lại:** ~14h (Phase 3.3 + 4).

---

## 6. Metrics Thành công — Actual vs Target

| Metric | Ban đầu | Target Phase 1 | Target Final | **Actual (2026-06-29)** | Status |
|---|---|---|---|---|---|
| Metadata coverage | 11.7% | 100% | 100% | **100%** (427/427) | ✅ |
| Detection in metadata | 0% | 93%+ | 93%+ | **88.3%** (377/427) | 🟡 Hơi thấp (50 frames YOLO không detect gì) |
| Scene populated | 70% | 100% | 100% | **100%** (427/427) | ✅ |
| Scene graph populated | 0% | 80%+ | 90%+ | **0%** (0/427) | 🔴 CHƯA ĐẠT (cần Phase 4.1) |
| Unique smoothed scenes | 82 | <50 | ~10 | **36** | 🟡 Chấp nhận được (TV cuts hợp lý) |
| Scene confidence ≥0.5 | 30% | 30% | 60%+ | **chưa đo lại** | — |
| OCR integrated | 0% | 0% | 100% | **100%** (3,139 regions) | ✅ |
| Caption available | 0% | 0% | 100% | **100%** (avg 46 chars) | ✅ |
| Caption empty | n/a | n/a | <5% | **0%** (0 empty) | ✅ |

---

## 7. Rủi ro và Biện pháp — Updated

| Rủi ro | Mức độ | Biện pháp | Status |
|---|---|---|---|
| Qwen2.5-VL quá chậm (7B params) | Trung bình | 4-bit quantize, batch processing, cache results | ⚠️ VẪN CÒN (chưa chạy end-to-end) |
| VRAM không đủ cho scene_graph + scene đồng thời | Cao | Sequential loading, offload model sau mỗi pass | ✅ Mitigated (code đã sẵn sàng) |
| Places365 domain mismatch | Trung bình | Fine-tune trên AIC domain hoặc dùng CLIP zero-shot scene | 🟡 Mitigated (smoothing giúp giảm noise) |
| Schema migration phá vỡ downstream | Thấp | Backward-compatible: chỉ thêm fields, không sửa/xóa | ✅ Mitigated (None khi chưa populate) |
| Caption model ITM variant gotcha | Trung bình | Documented `blip2-itm-vit-g` không thể caption (image-only) | ✅ Mitigated (dùng BLIP base, không phải BLIP-2 ITM) |

---

## 8. Outstanding Work — Cần làm tiếp

### Ngay lập tức (ưu tiên cao)
1. **Commit Phase 3.2** — captioning.py, test_captioning.py (mới), metadata.py, cli/main.py, settings.py, constants.py, test_metadata.py, test_cli_metadata.py (sửa).
2. **Phase 4.1 — Chạy Qwen scene graph end-to-end** trên K05_V004 → đóng P0 cuối cùng.

### Sau đó (ưu tiên trung bình)
3. **Phase 3.3 — Temporal context** (3h, low complexity).
4. **3.2.1 — Caption-based text search** integration trong `core/search.py`.

### Deferred (ưu tiên thấp)
5. **3.2.2 — Vietnamese captions** (cần model riêng, vd VinAI).
6. **Phase 4.2 — Incremental update mode**.
7. **Phase 4.3 — JSON Schema validation**.

---

> **Bước tiếp theo đề xuất:** Commit Phase 3.2 work (captioning.py + tests + plan update này) trước, sau đó bắt đầu **Phase 4.1** (chạy Qwen scene graph) để đóng P0 cuối cùng. Nếu không có GPU khả dụng ngay, ưu tiên **Phase 3.3** (temporal context) vì không cần GPU.