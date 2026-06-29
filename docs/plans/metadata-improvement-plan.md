# 📋 Kế hoạch Cải thiện Metadata — K05_V004

> **Ngày tạo:** 2026-06-29
> **Video mục tiêu phân tích:** K05_V004
> **Áp dụng cho:** Toàn bộ pipeline metadata AIC-26
> **Phase 1 Status:** ✅ HOÀN THÀNH (2026-06-29)
> **Phase 2 Status:** ✅ HOÀN THÀNH (2026-06-29)

---

## 1. Tổng quan Phân tích Hiện trạng

### 1.1 Dữ liệu Keyframes
| Chỉ số | Giá trị |
|---|---|
| Tổng keyframes | **427 frames** |
| Metadata sidecars đã tạo | **50 / 427** (11.7%) |
| Detection files đã tạo | **426 / 427** (99.8%) |
| Detection files có objects | **397 / 426** (93.2%) |

### 1.2 Các vấn đề NGHIÊM TRỌNG phát hiện

#### 🔴 P0 — Metadata Coverage cực thấp (11.7%)

**Mô tả:** Chỉ có **50/427 keyframes** (11.7%) có metadata sidecar JSON. Nghĩa là **377 frames hoàn toàn thiếu metadata** — không có scene classification, không có scene graph, detection không được tổng hợp vào sidecar.

**Nguyên nhân gốc:** Metadata extraction là **opt-in** (`extract_metadata=False` mặc định) trong `orchestrator.py:L300`. Khi chạy pipeline chuẩn, bước 6 bị skip. Ngoài ra, khi chạy với `--extract-metadata`, orchestrator truyền dummy lambdas:
```python
# orchestrator.py:L307-309 — DUMMY functions, không chạy model thực!
detector_fn=lambda p: None,
scene_fn=lambda img: None,
scene_graph_fn=lambda img: [],
```
→ Metadata sidecar được tạo nhưng với `detections: []`, `scene: null`, `scene_graph: []`.

#### 🔴 P0 — Detections hoàn toàn trống trong Metadata (100%)

**Mô tả:** Tất cả 50 metadata sidecars đều có `detections: []`. Trong khi đó, pipeline detection (Step 2) đã chạy thành công với **397/426 files có objects**. Dữ liệu detection đã tồn tại ở `data/detections/K05_V004/` nhưng **không được tổng hợp** vào metadata sidecar.

**Tác động:** Truy vấn kiểu "tìm frame có person ở top-left" qua metadata sẽ trả về kết quả rỗng, mặc dù data detection đã có sẵn.

#### 🔴 P0 — Scene Graph hoàn toàn trống (100%)

**Mô tả:** Tất cả 50 metadata sidecars có `scene_graph: []`. Qwen2.5-VL model **chưa bao giờ được invoke** vì orchestrator truyền dummy `scene_graph_fn=lambda img: []`.

**Tác động:** Truy vấn relational kiểu "person hold cup", "car park on street" hoàn toàn bất khả thi.

#### 🟡 P1 — Scene Classification confidence thấp (70%)

**Mô tả:** 35/50 frames (70%) có scene confidence < 0.5. Phân bố scene rất phân tán: **28 categories khác nhau cho chỉ 50 frames** — đáng ngờ. Các scene categories mâu thuẫn với nhau (vd: cùng video có `airport_terminal`, `hospital_room`, `candy_store`, `campsite`, `slum`).

**Phân tích:** Điều này cho thấy Places365 ResNet50 đang gán scene labels không nhất quán, có thể do:
1. Video chứa nhiều cảnh khác nhau thực sự (news broadcast)
2. Model không phù hợp với domain dữ liệu
3. Keyframes bị ảnh hưởng bởi chuyển cảnh, overlay text, hoặc chất lượng thấp

#### 🟡 P1 — Top-5 Scenes có duplicate category

**Mô tả:** Trong file `010.json`, top5 có 2 entries cho `"platform"` (indices khác nhau nhưng cùng label). Đây là lỗi từ Places365 label mapping — cùng leaf name nhưng paths khác nhau (`/p/platform/train_station_platform` vs `/p/platform/outdoor`).

#### 🟢 P2 — Thiếu các modalities bổ sung

Metadata schema hiện tại thiếu:
- **OCR text** — `data/ocr_results.csv` đã tồn tại nhưng chưa tích hợp vào metadata sidecar
- **Image captions** — Không có caption model (BLIP/BLIP-2)
- **Audio/ASR transcription** — Chưa có
- **Action recognition** — Chưa có
- **Temporal context** — Không liên kết frame trước/sau

---

## 2. Kế hoạch Cải thiện

### Phase 1: Khắc phục Khẩn cấp (P0) — ⏱️ 1-2 ngày

#### Task 1.1: Tích hợp Detection Data vào Metadata Sidecar
**File cần sửa:** `src/aic26/pipeline/metadata.py`

```
Hành động:
1. Sửa MetadataExtractor.process_keyframe() để đọc detection JSON 
   sẵn có từ data/detections/{video}/{frame}.json thay vì chạy lại detector
2. Tạo fallback detector_fn đọc từ pre-computed detection files
3. Merge detection data (bao gồm position enrichment) vào sidecar
```

**Pseudocode:**
```python
def _load_precomputed_detections(detection_dir: Path) -> DetectorFn:
    """Create a detector_fn that reads pre-computed detection JSONs."""
    def detector_fn(image_path: Path) -> dict | None:
        stem = image_path.stem
        video = image_path.parent.name
        det_path = detection_dir / video / f"{stem}.json"
        if det_path.exists():
            return json.loads(det_path.read_text())
        return None
    return detector_fn
```

#### Task 1.2: Wire Real Models vào Orchestrator
**File cần sửa:** `src/aic26/pipeline/orchestrator.py` (L306-310)

```
Hành động:
1. Thay dummy lambdas bằng real model callables
2. detector_fn → đọc từ data/detections/ (pre-computed)
3. scene_fn → Places365Classifier.from_weights()
4. scene_graph_fn → extract_scene_graph() với Qwen2.5-VL
```

#### Task 1.3: Chạy lại Metadata Extraction cho toàn bộ 427 frames
```bash
aic26 pipeline run-all --video-dir data/videos \
    --skip-detection --skip-ocr --skip-embeddings --skip-index \
    --extract-metadata
```
Hoặc tạo CLI command riêng:
```bash
aic26 pipeline extract-metadata --video K05_V004 --overwrite
```

---

### Phase 2: Nâng cao Scene Classification (P1) — ⏱️ 2-3 ngày

#### Task 2.1: De-duplicate Places365 Labels
**File cần sửa:** `src/aic26/pipeline/places365.py`

```
Hành động:
- Trong load_places365_labels(), de-duplicate leaf labels bằng cách 
  append parent path khi có collision
  Ví dụ: "platform" → "train_station_platform" vs "outdoor_platform"
```

#### Task 2.2: Scene Confidence Thresholding + Fallback
```
Hành động:
- Khi top-1 confidence < 0.3, đánh dấu scene là "uncertain"
- Aggregate top-5 vào categories nhóm (indoor/outdoor/transportation/...)
  để giảm noise
- Thêm field "scene_group" vào schema
```

#### Task 2.3: Scene Temporal Smoothing
```
Hành động:
- Các frames liền kề thường cùng scene → dùng sliding window 
  majority vote (window=5) để smooth scene labels
- Thêm "smoothed_scene" field vào sidecar
- Giảm đáng kể số categories bất thường (28 → ~5-8 expected)
```

---

### Phase 3: Bổ sung Modalities (P2) — ⏱️ 3-5 ngày

#### Task 3.1: Tích hợp OCR vào Metadata
```
Hành động:
- Parse ocr_results.csv → map text per (video, frame_idx)
- Thêm field "ocr_text": [{"text": str, "bbox": [...], "confidence": float}]
  vào KeyframeMetadata schema
- Hỗ trợ truy vấn "tìm frame có text chứa 'EXIT'"
```

#### Task 3.2: Image Captioning (BLIP-2)
```
Hành động:
- Thêm CaptionFn type alias vào metadata.py
- Sử dụng BLIP-2 (đã có model trong core/models.py) để generate caption
- Thêm field "caption": str vào sidecar
- Caption cho phép text search bằng ngôn ngữ tự nhiên mà EVA-02 
  embedding có thể miss
```

#### Task 3.3: Temporal Context
```
Hành động:
- Thêm fields:
    "prev_frame_idx": int | null,
    "next_frame_idx": int | null, 
    "temporal_segment_id": int  # nhóm frames liên tục cùng scene
- Cho phép navigation giữa các frames liên quan
```

---

### Phase 4: Tối ưu Pipeline (P2) — ⏱️ 2-3 ngày

#### Task 4.1: Batch Processing cho Scene Graph
```
Hành động:
- Scene graph extraction hiện xử lý từng frame riêng lẻ (~2-5s/frame)
- Implement batch inference cho Qwen2.5-VL (nếu VRAM cho phép)
- Ước tính: 427 frames × 3s = ~21 phút → batch 4 = ~5 phút
```

#### Task 4.2: Incremental Metadata Update
```
Hành động:
- Hiện tại process_video() skip file đã tồn tại khi overwrite=False
- Thêm mode "merge": đọc sidecar hiện có, chỉ cập nhật fields còn thiếu
  Ví dụ: sidecar có scene nhưng thiếu scene_graph → chỉ chạy scene_graph
- Giảm thời gian khi thêm modality mới
```

#### Task 4.3: Schema Validation + Migration
```
Hành động:
- Tạo JSON Schema chính thức cho metadata sidecar
- Script migrate cũ → schema mới (backward-compatible)
- CI test validate tất cả sidecars against schema
```

---

## 3. Schema Metadata Đề xuất (v2)

```json
{
  "version": "2.0",
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
  
  "caption": "A person holding a cup while standing in an airport terminal.",
  
  "temporal": {
    "prev_frame_idx": 41,
    "next_frame_idx": 43,
    "segment_id": 3,
    "timestamp_sec": 14.2
  }
}
```

---

## 4. Ưu tiên Thực hiện

| # | Task | Priority | Effort | Impact | Dependencies |
|---|---|---|---|---|---|
| 1.1 | Tích hợp Detection → Metadata | 🔴 P0 | 3h | Rất cao | Không |
| 1.2 | Wire real models vào orchestrator | 🔴 P0 | 4h | Rất cao | 1.1 |
| 1.3 | Chạy lại metadata extraction | 🔴 P0 | 1h | Rất cao | 1.2 |
| 2.1 | De-duplicate Places365 labels | 🟡 P1 | 2h | Trung bình | Không |
| 2.2 | Scene confidence thresholding | 🟡 P1 | 3h | Trung bình | 2.1 |
| 2.3 | Scene temporal smoothing | 🟡 P1 | 4h | Cao | 1.3 |
| 3.1 | Tích hợp OCR | 🟢 P2 | 3h | Cao | 1.3 |
| 3.2 | Image captioning (BLIP-2) | 🟢 P2 | 6h | Cao | GPU |
| 3.3 | Temporal context | 🟢 P2 | 3h | Trung bình | 1.3 |
| 4.1 | Batch scene graph | 🟢 P2 | 4h | Trung bình | GPU + Qwen |
| 4.2 | Incremental update mode | 🟢 P2 | 4h | Cao | 1.1 |
| 4.3 | Schema validation + migration | 🟢 P2 | 3h | Trung bình | 3.x |

**Tổng effort ước tính:** ~40h (~1 tuần full-time)

---

## 5. Metrics Thành công

| Metric | Hiện tại | Mục tiêu Phase 1 | Mục tiêu Final |
|---|---|---|---|
| Metadata coverage | 11.7% | **100%** | 100% |
| Detection in metadata | 0% | **93%+** | 93%+ |
| Scene graph populated | 0% | **80%+** | 90%+ |
| Scene confidence ≥0.5 | 30% | 30% | **60%+** |
| OCR integrated | 0% | 0% | **100%** |
| Caption available | 0% | 0% | **100%** |

---

## 6. Rủi ro và Biện pháp

| Rủi ro | Mức độ | Biện pháp |
|---|---|---|
| Qwen2.5-VL quá chậm (7B params) | Trung bình | 4-bit quantize, batch processing, cache results |
| VRAM không đủ cho scene_graph + scene đồng thời | Cao | Sequential loading, offload model sau mỗi pass |
| Places365 domain mismatch | Trung bình | Fine-tune trên AIC domain hoặc dùng CLIP zero-shot scene |
| Schema migration phá vỡ downstream | Thấp | Backward-compatible: chỉ thêm fields, không sửa/xóa |

---

> **Bước tiếp theo:** Bắt đầu với **Task 1.1** — tạo `_load_precomputed_detections()` trong `metadata.py` và wire vào orchestrator. Đây là fix nhanh nhất có tác động lớn nhất.
