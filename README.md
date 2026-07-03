<div align="center">

# AIC-26

**Hệ Thống Truy Xuất Video Thông Minh**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Code style: black](https://img.shields.io/badge/Code%20style-black-000000?style=flat-square)](https://github.com/psf/black)
[![Linting: Ruff](https://img.shields.io/badge/Linting-Ruff-FCC21B?style=flat-square)](https://github.com/astral-sh/ruff)

Tìm kiếm trong các bộ dữ liệu video lớn bằng ngôn ngữ tự nhiên, hình ảnh, hoặc chuỗi sự kiện theo thời gian - được hỗ trợ bởi EVA-02 CLIP embeddings và FAISS vector search.

[Bắt Đầu](#-bắt-đầu) · [Sử Dụng](#-sử-dụng) · [Kiến Trúc](#-kiến-trúc) · [API Reference](#-api-reference) · [Đóng Góp](#-đóng-góp)

</div>

---

## Tổng Quan

AIC-26 là một hệ thống truy xuất video đa phương thức được xây dựng cho cuộc thi AI Challenge. Hệ thống trích xuất keyframe từ video, lập chỉ mục chúng bằng các vision-language embeddings tiên tiến, và cung cấp tìm kiếm tương đồng nhanh thông qua REST API, giao diện desktop và CLI.

### Tính Năng Chính

| Tính năng | Mô tả |
|---|---|
| **Text-to-Video Search** | Tìm frame phù hợp với mô tả ngôn ngữ tự nhiên bằng tiếng Anh hoặc tiếng Việt |
| **Image-to-Video Search** | Truy xuất các frame có hình ảnh tương đồng bằng cách tải lên ảnh query |
| **TRAKE Temporal Search** | Xác định các chuỗi sự kiện có thứ tự trong video bằng dynamic programming |
| **Object Filtering** | Thu hẹp kết quả theo các đối tượng được phát hiện với ngưỡng confidence |
| **OCR Search** | Lập chỉ mục và tìm kiếm văn bản xuất hiện trong các frame video |
| **Dual-Embedding Retrieval** | Kết hợp EVA-02 (w=0.6) + CLIP-32 (w=0.4) + BM25 captions (w=0.2) — 3-way weighted blend |
| **Metadata Sidecars** | Enrich mỗi keyframe với bbox position (3×3 grid), Places365 scene + scene_group, OCR text và BLIP caption — xem [docs/specs/metadata-extraction.md](docs/specs/metadata-extraction.md) |
| **Client-side Filtering** | Giao diện hỗ trợ lọc Metadata (OCR, Scene, Caption) tức thì bằng `Fuse.js`, tối ưu hóa trải nghiệm truy vấn |

### Công Nghệ Sử Dụng

| Thành phần | Công nghệ | Mục đích |
|---|---|---|
| Vision-Language Model | EVA-02 CLIP (768-dim) | Embedding text và hình ảnh (stream chính, eager-load) |
| Dual-Stream Encoder | CLIP ViT-B-32 (512-dim) | Stream thứ hai cho weighted blend (eager-load) |
| Text Retrieval | BM25 (rank_bm25) | Keyword match trên captions (stream thứ ba) |
| Vector Search | FAISS IndexFlatIP × 2 streams | Tìm kiếm tương đồng dưới một giây |
| Temporal Reasoning | TRAKE (custom DP) | Truy xuất sự kiện có thứ tự |
| Object Detection | **YOLOv26x** (COCO 80 classes) | Bbox + label cho metadata sidecar |
| Caption Generation | BLIP image captioning | English caption cho mỗi keyframe |
| Reranking | BLIP ITM (lazy-load) | Cải thiện độ chính xác TRAKE |
| OCR | EasyOCR (pre-extracted CSV) | Trích xuất văn bản tiếng Việt/tiếng Anh |
| Keyframe Extraction | DINOv2 + cosine similarity | Phát hiện thay đổi cảnh |
| API Server | Flask + Flask-CORS (threaded) | REST endpoints |
| Web UI Filtering | Fuse.js (client-side) | Fuzzy filter OCR/Scene/Caption tức thì |
| Desktop GUI | CustomTkinter | Giao diện desktop |

---

## Bắt Đầu

### Yêu Cầu

- Python 3.10 trở lên
- RAM tối thiểu 16 GB (khuyến nghị 32 GB)
- 50 GB dung lượng ổ đĩa trống cho dữ liệu video và embeddings
- GPU hỗ trợ CUDA (không bắt buộc nhưng khuyến nghị mạnh)

### Cài Đặt

**1. Clone repository**

```bash
git clone https://github.com/huudangg0204/AIC-26.git
cd aic26
```

**2. Cài đặt PyTorch** (chọn lệnh phù hợp với môi trường của bạn tại [pytorch.org](https://pytorch.org/get-started/locally/))

```bash
# Chỉ CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**3. Cài đặt package**

```bash
pip install -e .

# Với hỗ trợ GPU cho FAISS
pip install -e ".[gpu]"

# Với các công cụ phát triển
pip install -e ".[dev]"
```

**4. Kiểm tra cài đặt**

```bash
aic26 check
```

### Cấu Hình

Sao chép file môi trường mẫu và chỉnh sửa theo nhu cầu:

```bash
cp .env.example .env
```

Các cài đặt chính trong `.env`:

```ini
# Đường dẫn dữ liệu
EMBEDDING_DIR=data/embeddings
KEYFRAMES_DIR=data/keyframes
DB_PATH=data/faiss_db.pkl

# Mặc định tìm kiếm
DEFAULT_TOP_K=100
DEFAULT_SCORE_THRESHOLD=0.2

# Metadata sidecars (extract-metadata, merge-ocr, merge-captions)
METADATA_DIR=data/metadata
CAPTION_MODEL_NAME=Salesforce/blip-image-captioning-base
CAPTION_MAX_NEW_TOKENS=30
PLACES365_MODEL_PATH=models/places365/resnet50_places365.pth.tar

# Server
HOST=0.0.0.0
PORT=5000
DEBUG=false

# Logging
LOG_LEVEL=INFO
```

Tất cả cài đặt cũng có thể được truyền qua biến môi trường. Chạy `aic26 config --show` để xem các giá trị đang hoạt động.

---

## Sử Dụng

### 1. Chuẩn Bị Dữ Liệu

Đặt các file video vào `data/videos/` và chạy toàn bộ pipeline xử lý:

```bash
aic26 pipeline run-all --video-dir data/videos
```

Pipeline thực hiện **7 bước tuần tự** với GPU memory cleanup giữa mỗi bước:

```
Videos → Keyframes → Detection (YOLOv26) → Embeddings (EVA-02 + CLIP-32)
       → FAISS Index → Metadata (merge OCR + Captions) → Ready
```

> **Lưu ý VRAM**: Pipeline được tối ưu cho GPU 4GB (RTX 3050). Mỗi model được load/unload tuần tự
> với `torch.cuda.empty_cache()` giữa các bước để tránh OOM.

Từng bước có thể chạy riêng lẻ hoặc bỏ qua:

```bash
# Chỉ trích xuất keyframe
aic26 pipeline extract-keyframes --input data/videos --output data/keyframes

# Trích xuất EVA-02 embeddings (batch size 40, khuyến nghị dùng GPU)
aic26 pipeline extract-embeddings --input data/keyframes --batch-size 40

# Xây dựng FAISS index
aic26 pipeline build-index --input data/embeddings

# Bỏ qua detection và OCR để chạy nhanh hơn
aic26 pipeline run-all --video-dir data/videos --skip-detection --skip-ocr
```

### 2. Khởi Động API Server

```bash
aic26 serve api --port 5000
```

*Lưu ý: Server eager-load **EVA-02 CLIP + CLIP ViT-B-32** (~22s startup). Metadata được load từ pre-built pickle cache (~48MB, < 1s). BLIP ITM và Faster R-CNN chỉ lazy-load khi cần. Chạy `py -3.10 scripts/build_meta_cache.py` sau mỗi lần pipeline chạy xong để cập nhật cache.*

Mở [http://localhost:5000](http://localhost:5000) để sử dụng giao diện web Multi-modal Query Builder tích hợp sẵn (Hỗ trợ Fuse.js Filtering, TRAKE, và Modal Metadata Inspector).

### 3. Khởi Chạy Desktop GUI

```bash
aic26 serve gui
```

### 4. Tìm Kiếm Qua CLI

```bash
# Tìm kiếm bằng text
aic26 search text --query "a red car driving on a highway" --top-k 20

# Tìm kiếm theo thời gian (TRAKE)
aic26 search temporal \
  --events "a player dribbling the ball" \
  --events "a player shooting a basketball" \
  --events "the crowd cheering" \
  --mode highest
```

### 5. Metadata Sidecars

Mỗi keyframe được enrich bằng một JSON sidecar tại `data/metadata/{video}/{frame_idx}.json`, kết hợp 5 tín hiệu trực giao để hỗ trợ query lọc theo **vị trí object**, **scene**, **quan hệ giữa object**, **OCR text** và **BLIP caption**. Triển khai theo [docs/specs/metadata-extraction.md](docs/specs/metadata-extraction.md) (kết hợp SnapSeek 3.0 §3.5 + lifeXplore §3).

```bash
# Trích xuất toàn bộ metadata (bbox + Places365 + scene graph qua Qwen2.5-VL)
aic26 pipeline extract-metadata -i data/keyframes -o data/metadata [--skip-scene-graph]

# Verify một keyframe cụ thể
aic26 pipeline verify-metadata --video K05_V004 --frame-idx 42

# Thống kê coverage
aic26 pipeline metadata-stats -i data/metadata

# Gộp OCR/EasyOCR text regions vào sidecar (chạy sau extract-metadata)
aic26 pipeline merge-ocr --input data/ocr_results.csv --metadata-dir data/metadata

# Tạo BLIP captions cho keyframes
aic26 pipeline merge-captions --metadata-dir data/metadata
```

**Sidecar schema (v1)**:

```json
{
  "video": "K05_V004",
  "frame_idx": 42,
  "detections": [
    {"label": "Person", "score": 0.92, "bbox": [10, 20, 100, 200], "position": "top-left"}
  ],
  "scene": {
    "category": "airport_terminal", "confidence": 0.633,
    "top5": [["airport_terminal", 0.633], ["indoor", 0.185]],
    "smoothed_scene": "airport_terminal", "scene_group": "indoor_public"
  },
  "scene_graph": [{"s": "person", "p": "hold", "o": "cup", "confidence": 0.9}],
  "ocr_text": [{"text": "Kết quả bốc thăm", "bbox": [12, 230, 410, 280], "confidence": 0.84}],
  "caption": "a man holding a trophy on stage"
}
```

**Quy ước**: `detections = []` khi YOLO không tìm được object; `scene` / `scene_graph` là `null` khi model bị skip, `[]` / `{...}` khi chạy nhưng trống.

**Pipeline Models**:
- **YOLOv26x** (`pipeline/detection.py`) — 80 COCO classes, `num_workers=1` cho low VRAM.
- **BLIP image captioning** (`pipeline/captioning.py`) — tiếng Anh, ~446M params.
- **EasyOCR** (pre-extracted CSV) — merge từ `data/raw/ocr_results.csv`.

**Serving Models**:
- **EVA-02 CLIP** — eager-loaded at startup (~1.7GB VRAM), text + image encoding.
- **CLIP ViT-B-32** — eager-loaded at startup (~0.5GB VRAM), dual-stream search.
- **BLIP ITM** — lazy-loaded for TRAKE reranking only (~0.9GB VRAM).
- **Faster R-CNN** — lazy fallback, only if metadata missing (~0.2GB VRAM).

**Kết quả trên K05_V004** (theo [docs/plans/metadata-improvement-plan.md](docs/plans/metadata-improvement-plan.md)):

| Signal | Coverage | Notes |
|---|---|---|
| Detections (w/ position) | 88.3% | 377/427 sidecars |
| Scene (Places365) | 100% | Có temporal smoothing sliding window |
| Scene group (13 nhóm) | 100% | `outdoor_urban`, `indoor_public`, ... |
| OCR text | 100% | 3,139 regions merged |
| Caption (English, BLIP) | 100% | 46 chars/caption avg |
| Scene graph (Qwen) | 0% | Phase 4.1 — sẽ chạy sau |

**Client-side filtering**: Web UI (`src/aic26/api/templates/index.html`) dùng Fuse.js để lọc metadata tức thì trên các kết quả trả về — không phải query lại server. Modal "Metadata Inspector" cho phép click vào từng frame để xem sidecar đầy đủ.

---

## Kiến Trúc

```
┌─────────────────────────────────────────────────────────────┐
│                        Interfaces                           │
│     REST API (Flask, threaded)  ·  Desktop GUI  ·  CLI      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              EVA02ImageRetrieval (Facade)                    │
│    Metadata Cache (shared pickle, ~48MB, 41675 entries)     │
└──┬──────────┬──────────┬──────────┬──────────┬─────────────┘
   │          │          │          │          │
┌──▼──┐  ┌───▼───┐  ┌───▼──┐  ┌───▼───┐  ┌───▼────┐
│Model│  │Embed- │  │Search│  │ TRAKE │  │Exporter│
│Reg- │  │ding   │  │Engine│  │Engine │  │        │
│istry│  │Index  │  │(3-way│  │(batch │  │        │
│     │  │(dual) │  │blend)│  │rerank)│  │        │
│     │  └───┬───┘  └───┬──┘  └───┬───┘  └────────┘
└──┬──┘      │          │          │
   │    ┌────▼──────────▼──────────▼──────────────────────┐
   └───▶│              Core Models (GPU)                  │
        │  EVA-02 (eager)  ·  CLIP-32 (eager)             │
        │  BLIP ITM (lazy) ·  Faster R-CNN (lazy fallback)│
        └─────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│             Data Pipeline (GPU-safe, sequential)            │
│  Keyframes → YOLOv26 Detection → EVA-02 + CLIP-32 Embed   │
│       → FAISS Index → Metadata (OCR merge + Captions)      │
│  [GPU cleanup between each step for 4GB VRAM safety]       │
└────────────────────────────────────────────────────────────┘
```

### Cấu Trúc Dự Án

```
aic26/
├── src/aic26/
│   ├── config/          # Pydantic settings và hằng số
│   ├── core/            # Retrieval engine
│   │   ├── models.py    # ModelRegistry (lazy-load: EVA-02 eager, others lazy)
│   │   ├── embeddings.py# EmbeddingIndex (dual: EVA-02 768d + CLIP-32 512d)
│   │   ├── search.py    # 3-way blend search (EVA-02 + CLIP-32 + BM25)
│   │   ├── trake.py     # Temporal retrieval (thuật toán TRAKE)
│   │   ├── rerank.py    # BLIP ITM reranker
│   │   ├── database.py  # Persistent query store (FAISS + pickle)
│   │   ├── export.py    # CSV submission exporter
│   │   └── retrieval.py # Public facade (EVA02ImageRetrieval)
│   ├── pipeline/        # Xử lý dữ liệu (GPU-safe sequential)
│   │   ├── keyframe.py  # Trích xuất keyframe bằng DINOv2
│   │   ├── detection.py # Object detection bằng YOLOv26x (num_workers=1)
│   │   ├── ocr.py       # Trích xuất text bằng EasyOCR
│   │   ├── embedding.py # Dual embedding (EVA-02 768d + CLIP-32 512d)
│   │   ├── metadata.py  # Metadata sidecar builder (merge OCR + captions)
│   │   ├── captioning.py# BLIP caption generation
│   │   ├── indexer.py   # FAISS index builder
│   │   └── orchestrator.py # Điều phối pipeline
│   ├── api/             # Flask REST API
│   │   ├── server.py    # App factory
│   │   ├── routes/      # search, trake, export, video, system
│   │   └── middleware/  # Xử lý lỗi
│   ├── gui/             # Desktop GUI (CustomTkinter)
│   ├── cli/             # CLI entry point (Click)
│   └── utils/           # Tiện ích logging, file, frame, media
├── data/
│   ├── videos/          # File video đầu vào
│   ├── keyframes/       # Ảnh keyframe đã trích xuất
│   ├── embeddings/      # File embedding .npy đã tính trước
│   ├── detections/      # JSON kết quả detection (YOLOv26)
│   ├── metadata/        # JSON sidecars (detections + OCR + captions)
│   ├── raw/             # Dữ liệu thô (ocr_results.csv)
│   └── submissions/     # Output CSV để nộp bài
├── tests/
├── docker/
├── configs/
└── pyproject.toml
```

---

## API Reference

REST API khả dụng tại `http://localhost:5000/api`. Tất cả endpoint đều nhận và trả về JSON trừ khi có ghi chú khác.

### Tìm Kiếm

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/api/search_text` | Tìm kiếm bằng text query |
| `POST` | `/api/search_image` | Tìm kiếm bằng đường dẫn ảnh |
| `POST` | `/api/upload_image` | Tải ảnh lên và tìm kiếm |
| `POST` | `/api/search_text_with_image` | Tìm kiếm kết hợp text + ảnh |

**Ví dụ tìm kiếm bằng text:**

```bash
curl -X POST http://localhost:5000/api/search_text \
  -H "Content-Type: application/json" \
  -d '{
    "query": "a person riding a bicycle on a sunny day",
    "top_k": 20,
    "objects": "bicycle,person",
    "threshold": 0.5
  }'
```

```json
{
  "elapsed": 0.412,
  "count": 20,
  "results": [
    {
      "image_path": "data/keyframes/L01_V001/042.jpg",
      "video_name": "L01_V001",
      "similarity": 0.3821,
      "frame_idx": "42",
      "time_str": "1:24",
      "fps": 30.0,
      "watch_url": ""
    }
  ]
}
```

### TRAKE - Tìm Kiếm Theo Thời Gian

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/api/trake` | Tìm chuỗi sự kiện có thứ tự |

```bash
curl -X POST http://localhost:5000/api/trake \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      "a player dribbling a basketball",
      "a player shooting*",
      "the crowd cheering"
    ],
    "top_k": 200,
    "option": "Highest"
  }'
```

> Các sự kiện kết thúc bằng `*` sẽ được rerank bằng BLIP ITM để tăng độ chính xác.

### Export

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/api/export_results` | Export kết quả tìm kiếm dạng CSV |
| `POST` | `/api/export_trake_results` | Export kết quả TRAKE dạng CSV |
| `POST` | `/api/export_submission` | Tạo CSV nộp bài xung quanh một frame |

### System

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/api/stats` | Thống kê index và database |
| `GET` | `/api/stored_queries` | Danh sách lịch sử query đã lưu |
| `GET` | `/api/reload_query/<idx>` | Tải lại kết quả từ query trước đó |
| `POST` | `/api/group_by_video` | Nhóm kết quả theo video |
| `GET` | `/health` | Kiểm tra trạng thái hệ thống |

---

## Thuật Toán TRAKE

TRAKE (Temporal Retrieval and Alignment of Key Events) tìm các chuỗi keyframe khớp với danh sách mô tả sự kiện theo đúng thứ tự thời gian.

**Hai chiến lược có sẵn:**

`Closest` - tối thiểu hóa tổng khoảng cách frame giữa các sự kiện (greedy DP). Tốt nhất khi các sự kiện xảy ra gần nhau trong thời gian.

`Highest` - tối đa hóa tổng điểm similarity trên các sự kiện (score-accumulation DP). Tốt nhất để tìm chuỗi có độ liên quan tổng thể cao nhất.

**Cách hoạt động:**

1. Mỗi sự kiện được tìm kiếm độc lập để lấy các frame ứng viên
2. Các sự kiện kết thúc bằng `*` được rerank bằng BLIP ITM để tăng độ chính xác
3. Với mỗi video ứng viên, dynamic programming xây dựng các chuỗi có thứ tự
4. Kết quả được xếp hạng theo điểm và trả về

---

## Chiến Thuật Truy Vấn (Query Strategies)

### Cách hệ thống xử lý một truy vấn

```
User query "a man talking on stage"
     │
     ├─→ EVA-02 encode (768d) ─→ cosine sim vs 41675 vectors ─→ top pool (w=0.6)
     ├─→ CLIP-32 encode (512d) ─→ cosine sim vs 41675 vectors ─→ top pool (w=0.4)
     ├─→ BM25 keyword match on captions ─→ top pool (w=0.2)
     │
     └─→ Union candidates → Weighted blend → Top-K results
                                                  │
                                            ┌─────▼─────┐
                                            │ Metadata   │
                                            │ attach:    │
                                            │ OCR, scene,│
                                            │ caption,   │
                                            │ detections │
                                            └─────┬─────┘
                                                  │
                                           Fuse.js filter
                                           (client-side)
```

### Chiến thuật 1: Text Search đơn giản

Dùng cho câu hỏi mô tả trực tiếp cảnh vật.

```
Query: "a woman wearing a red dress walking on the street"
Top-K: 50-100
Objects: (trống)
```

> **Mẹo**: Viết query bằng tiếng Anh, mô tả cụ thể. CLIP hiểu tốt nhất các câu có cấu trúc `<subject> <action> <context>`.

### Chiến thuật 2: Text + Object Filter

Dùng khi biết rõ đối tượng xuất hiện trong frame.

```
Query: "traffic accident on highway"
Objects: "car, truck"          // Giữ frame có ít nhất 1 object
Threshold: 0.5                 // Ngưỡng confidence
```

```
Objects: "person"              // Loại bỏ frame có người
Threshold: -0.5                // Threshold âm = exclude mode
```

> **Lưu ý**: Object filter sử dụng precomputed YOLOv26 detections từ metadata (không chạy model realtime).

### Chiến thuật 3: Fuzzy Filter (Fuse.js) — Lọc OCR/Caption/Scene

Sau khi nhận kết quả, dùng thanh filter trên cùng để lọc theo **OCR text**, **caption**, hoặc **scene**.

```
Ví dụ: Search "a news presenter" → 100 results
Fuse.js filter: "HTV"     → Chỉ giữ frame có chữ "HTV" trên màn hình
Fuse.js filter: "TP.HCM"  → Chỉ giữ frame có OCR chứa "TP.HCM"
Fuse.js filter: "sunset"  → Chỉ giữ frame có caption chứa "sunset"
```

> **Đây là tính năng mạnh nhất** cho cuộc thi AIC — kết hợp visual search + OCR text filtering.

### Chiến thuật 4: TRAKE — Temporal Search

Dùng khi cần tìm **chuỗi sự kiện theo thứ tự thời gian** trong cùng 1 video.

```
Events:
  E1: "a news anchor sitting at desk"
  E2: "a reporter standing outside"
  E3: "aerial view of a city"

Mode: Highest (tối đa similarity tổng thể)
Top-K: 200, Candidates: 200
```

> **Tip**: Thêm `*` cuối event (e.g. `"a reporter standing outside*"`) để trigger BLIP ITM reranking cho event đó — tăng precision nhưng chậm hơn (~3-5s/event).

### Chiến thuật 5: Image Search

Dùng khi có ảnh mẫu muốn tìm frame tương tự.

```
Upload ảnh query → Hệ thống encode bằng EVA-02
→ Cosine similarity vs toàn bộ index
→ Trả về top-K frames tương tự nhất
```

### Tips nâng cao

| # | Tip | Lý do |
|---|-----|-------|
| 1 | **Viết query tiếng Anh, ngắn gọn** | EVA-02 + CLIP trained trên English text |
| 2 | **Dùng `top_k=100`** rồi lọc Fuse.js | Server trả kết quả trong ~0.1s, lọc client-side tức thì |
| 3 | **Kết hợp Text Search + Fuse.js OCR** | Visual search tìm cảnh, OCR filter tìm text cụ thể |
| 4 | **Dùng TRAKE cho câu hỏi temporal** | "Trước cảnh X có cảnh gì?" → TRAKE tìm chuỗi |
| 5 | **Object filter exclude mode** | `threshold: -0.5` loại bỏ frame có object không mong muốn |
| 6 | **Xem modal Metadata Inspector** | Click vào kết quả để xem OCR text, detections, caption đầy đủ |

### Performance Benchmarks (RTX 3050 4GB, 41,675 keyframes)

| Thao tác | Thời gian | Ghi chú |
|----------|-----------|---------|
| Text Search (top_k=100) | **~0.1s** | 3-way blend: EVA-02 + CLIP-32 + BM25 |
| Image Search (top_k=100) | **~0.3s** | EVA-02 encode + cosine sim |
| TRAKE (3 events) | **~1-5s** | Phụ thuộc số events và reranking |
| Fuse.js Filter | **~0ms** | Client-side, không call server |
| Metadata attach | **~0ms** | Shared pickle cache (41,675 entries) |
| Server startup | **~22s** | EVA-02 + CLIP-32 eager-load + pickle cache |
| VRAM usage (idle) | **~2.5GB** | EVA-02 + CLIP-32 (both eager-loaded) |

---

## Docker (Đồng Bộ Đa Nền Tảng)

Dự án hỗ trợ Docker GPU để đảm bảo sự đồng bộ tuyệt đối trên tất cả các thiết bị trong team (Windows & Linux) sử dụng card đồ họa NVIDIA.

### Yêu Cầu Hệ Thống

**Đối với Windows 10/11:**
1. Cài đặt **NVIDIA Driver** mới nhất cho Windows.
2. Cài đặt **Docker Desktop** và bật tính năng **WSL 2 based engine**. (GPU sẽ tự động được passthrough vào container).

**Đối với Linux (Ubuntu/Debian):**
1. Cài đặt **NVIDIA Driver**.
2. Cài đặt **Docker Engine** và **Docker Compose**.
3. Cài đặt [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (`nvidia-docker2`).

### Cách Chạy Toàn Bộ Pipeline

Khi có video mới thả vào thư mục `data/videos`, bạn chỉ cần chạy một lệnh duy nhất từ thư mục gốc của dự án:

```bash
docker compose -f docker/docker-compose.yml --profile pipeline up --build
```

Lệnh này sẽ tự động:
1. Chạy module dọn dẹp và chuẩn hóa định dạng video (từ AV1, MKV,... sang MP4 H.264).
2. Chạy toàn bộ 5 bước của Pipeline AIC-26 (trích xuất keyframe, detection, OCR, EVA-02 embedding, xây dựng FAISS Index) hoàn toàn bằng GPU.

### Khởi Chạy API Server

Để chạy REST API server phục vụ truy xuất web/desktop (sử dụng GPU):

```bash
docker compose -f docker/docker-compose.yml --profile gpu up --build -d
```
API sẽ hoạt động tại `http://localhost:5000/api`.

---

## Phát Triển

### Thiết Lập

```bash
pip install -e ".[dev]"
pre-commit install
```

### Chạy Tests

```bash
# Tất cả tests
pytest

# Bỏ qua các test chậm (yêu cầu GPU hoặc model lớn)
pytest -m "not slow and not gpu"

# Với báo cáo coverage
pytest --cov=aic26 --cov-report=html
```

### Chất Lượng Code

```bash
# Định dạng code
black src/ tests/

# Lint (tự động sửa)
ruff check --fix src/

# Kiểm tra kiểu dữ liệu
mypy src/
```

### Build Tài Liệu

```bash
pip install -e ".[docs]"
mkdocs serve
```

---

## Đóng Góp

1. Fork repository và tạo feature branch từ `main`
2. Viết hoặc cập nhật tests cho bất kỳ hành vi mới nào
3. Đảm bảo tất cả kiểm tra đều vượt qua: `pre-commit run --all-files`
4. Mở pull request với mô tả rõ ràng về các thay đổi

Vui lòng đọc [CONTRIBUTING.md](CONTRIBUTING.md) để biết hướng dẫn về code style và quy ước đặt tên branch.

---

## Changelog

Xem [CHANGELOG.md](CHANGELOG.md) để biết lịch sử phiên bản.

---

## Giấy Phép

Dự án này được cấp phép theo MIT License - xem [LICENSE](LICENSE) để biết chi tiết.