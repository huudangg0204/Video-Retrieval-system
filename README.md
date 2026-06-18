<div align="center">

# AIC-26 — Video Retrieval System

**Hệ Thống Truy Xuất Video Thông Minh (MemoriEase 3.0)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.13-005571?style=flat-square&logo=elasticsearch)](https://www.elastic.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

Tìm kiếm trong cơ sở dữ liệu video lớn bằng ngôn ngữ tự nhiên, hình ảnh, hoặc chuỗi sự kiện theo thời gian — được hỗ trợ bởi **EVA-02 CLIP + BLIP-ITM dual embeddings** và **Elasticsearch** vector search.

[Bắt Đầu](#-bắt-đầu) · [Ingest Data](#-ingest-data) · [CLI Usage](#-cli-usage) · [API Reference](#-api-reference) · [Kiến Trúc](#-kiến-trúc)

</div>

---

## Tổng Quan

AIC-26 (MemoriEase 3.0) là hệ thống truy xuất video đa phương thức xây dựng cho cuộc thi AI Challenge 2026. Hệ thống trích xuất keyframe từ video, lập chỉ mục bằng dual embedding (EVA-02 + BLIP-ITM), lưu vào Elasticsearch, và cung cấp tìm kiếm tương đồng nhanh qua REST API và Web UI.

### Tính Năng Chính

| Tính năng | Mô tả |
|---|---|
| **Text Search** | Tìm frame bằng mô tả ngôn ngữ tự nhiên (English/Vietnamese) |
| **Image Search** | Truy xuất frame tương đồng bằng ảnh query |
| **TRAKE Temporal** | Xác định chuỗi sự kiện có thứ tự từ video bằng dynamic programming |
| **Dual Embedding** | EVA-02 (60%) + BLIP-ITM (40%) weighted kNN search |
| **Object Filtering** | Lọc kết quả theo đối tượng phát hiện (600+ classes) |
| **OCR Search** | Lập chỉ mục và tìm kiếm văn bản xuất hiện trong frame |
| **Web Ingest UI** | Upload video hoặc keyframe ZIP qua giao diện web, xem tiến độ real-time |

### Công Nghệ Sử Dụng

| Thành phần | Công nghệ | Mục đích |
|---|---|---|
| Vision-Language | EVA-02 CLIP Large (768-dim) | Image & text embedding |
| Reranker | BLIP-ITM Base COCO | Dual embedding & reranking |
| Vector Search | Elasticsearch 8.13 kNN | Scalable similarity search |
| Temporal Reasoning | TRAKE (custom DP) | Event sequence retrieval |
| Object Detection | Faster R-CNN | Frame annotation (600+ classes) |
| OCR | PaddleOCR | Text extraction (Vi/En) |
| Keyframe Extraction | DINOv2 + cosine similarity | Scene change detection |
| API Server | Flask 3 + Flask-CORS | REST endpoints |
| GPU | RTX 3050 Laptop (4GB) | ~1.25GB VRAM usage |

---

## Bắt Đầu

### Yêu Cầu

- Python 3.10+
- RAM ≥ 16 GB
- GPU CUDA (khuyến nghị, RTX 3050+ đủ dùng)
- Docker Desktop (cho Elasticsearch)
- 50 GB dung lượng ổ đĩa

### Cài Đặt

**1. Clone repository**

```bash
git clone https://github.com/huudangg0204/AIC-26.git
cd AIC-26
```

**2. Cài đặt PyTorch** (chọn lệnh phù hợp tại [pytorch.org](https://pytorch.org/get-started/locally/))

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**3. Cài đặt package**

```bash
py -3.10 -m pip install -e .
```

**4. Khởi động Elasticsearch**

```bash
docker compose -f docker/docker-compose.yml up elasticsearch -d
```

**5. Tạo ES index**

```bash
py -3.10 -m aic26.cli.main pipeline init-es-index
```

**6. Khởi động Web Server**

```bash
py -3.10 -m aic26.cli.main serve web
# hoặc
py -3.10 -c "
import sys; sys.path.insert(0,'src')
from aic26.api.server import create_app
from aic26.config import settings
create_app(settings).run(host='0.0.0.0', port=5000)
"
```

Mở trình duyệt: **http://localhost:5000**

---

## Ingest Data

AIC-26 hỗ trợ 2 cách nhập dữ liệu:

### Mode A — Keyframe Folder / ZIP (Kaggle dataset)

```
input/
├── L01_V001/        ← tên video
│   ├── 000.jpg
│   ├── 001.jpg
│   └── ...
└── L01_V002/
    └── ...
```

```bash
# Từ folder đã extract
py -3.10 -m aic26.cli.main ingest keyframes --input data/keyframes/

# Từ file .zip (Kaggle dataset)
py -3.10 -m aic26.cli.main ingest keyframes --input dataset.zip
```

### Mode B — Raw Video Files

```bash
py -3.10 -m aic26.cli.main ingest videos --input data/videos/
```

### Auto-detect

```bash
py -3.10 -m aic26.cli.main ingest auto --input data/
```

### Via Web UI

1. Mở **http://localhost:5000**
2. Kéo thả file `.zip` hoặc video vào drop zone
3. Chọn options (Skip OCR, Skip Detection nếu cần nhanh hơn)
4. Click **▶ Start Ingestion**
5. Xem progress bar real-time

### Options

| Flag | Mô tả |
|---|---|
| `--skip-detection` | Bỏ qua object detection |
| `--skip-ocr` | Bỏ qua OCR |
| `--skip-es` | Chỉ extract embedding, không index ES |
| `--force-recreate` | Xóa và tạo lại ES index |
| `--no-blip` | Chỉ dùng EVA-02 (không BLIP, nhanh hơn) |
| `--batch-size N` | GPU batch size (default: 40) |

---

## CLI Usage

```bash
# Alias (thêm vào PowerShell profile)
function aic26 { py -3.10 -m aic26.cli.main @args }

# Kiểm tra kết nối Elasticsearch
aic26 check-es

# Tạo ES index
aic26 pipeline init-es-index

# Index embeddings vào ES
aic26 pipeline index-to-es --embedding-dir data/embeddings

# Ingest keyframe dataset
aic26 ingest keyframes --input data/keyframes/ --skip-ocr

# Ingest video files
aic26 ingest videos --input data/videos/

# Text search
aic26 search text --query "a person riding a bicycle"

# Xem tất cả commands
aic26 --help
aic26 ingest --help
aic26 pipeline --help
```

---

## API Reference

Server chạy tại `http://localhost:5000`

### Search

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/api/stats` | Thống kê index ES |
| `POST` | `/api/search_text` | Text-based search |
| `POST` | `/api/upload_image` | Image-based search |
| `POST` | `/api/search_text_with_image` | Text + image reranking |
| `POST` | `/api/trake` | TRAKE temporal search |
| `POST` | `/api/group_by_video` | Group kết quả theo video |

### Ingest

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/api/ingest/upload` | Upload file + bắt đầu ingestion |
| `GET` | `/api/ingest/status/<job_id>` | Poll tiến độ job |
| `GET` | `/api/ingest/jobs` | Danh sách tất cả jobs |
| `POST` | `/api/ingest/cancel/<job_id>` | Huỷ job |

### Export

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/api/export_results` | Xuất kết quả ra CSV |
| `POST` | `/api/export_submission` | Tạo file submission |
| `POST` | `/api/export_trake_results` | Xuất TRAKE results |

### Ví dụ

```bash
# Text search
curl -X POST http://localhost:5000/api/search_text \
  -H "Content-Type: application/json" \
  -d '{"query": "a person riding a bicycle", "top_k": 10}'

# Upload file để ingest
curl -X POST http://localhost:5000/api/ingest/upload \
  -F "file=@keyframes.zip" \
  -F "skip_ocr=true" \
  -F "batch_size=40"

# Poll job status
curl http://localhost:5000/api/ingest/status/<job_id>
```

---

## Kiến Trúc

### MemoriEase 3.0 Architecture

```
┌─────────────────── Offline Pipeline ───────────────────────┐
│                                                             │
│  Video/Keyframes → KeyframeExtractor → ObjectDetector       │
│                          ↓                                  │
│               EmbeddingExtractor (EVA-02)                   │
│                          ↓                                  │
│              ElasticsearchIndexer (bulk)                    │
│                          ↓                                  │
│          Elasticsearch 8.13 (aic26_keyframes)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────── Online Search ──────────────────────────┐
│                                                             │
│  Query (text/image)                                         │
│       ↓                                                     │
│  EmbeddingEncoder (EVA-02 + BLIP-ITM)                       │
│       ↓                                                     │
│  ES kNN Search (dual embedding, weighted avg)               │
│       ↓  (60% EVA-02 + 40% BLIP-ITM)                       │
│  Reranking → TRAKE (temporal) → Results                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Cấu Trúc Thư Mục

```
AIC-26/
├── src/aic26/
│   ├── api/                  # Flask server + routes
│   │   ├── routes/
│   │   │   ├── search.py     # Search endpoints
│   │   │   ├── ingest.py     # Upload & ingest endpoints (NEW)
│   │   │   ├── trake.py      # TRAKE temporal search
│   │   │   ├── export.py     # CSV export
│   │   │   └── video.py      # Frame viewer
│   │   ├── templates/
│   │   │   └── index.html    # Dark UI (NEW)
│   │   └── server.py         # Flask app factory
│   ├── cli/
│   │   └── main.py           # CLI commands (ingest, pipeline, search)
│   ├── config/
│   │   └── settings.py       # App configuration
│   ├── core/
│   │   ├── elasticsearch_client.py  # ES wrapper (NEW)
│   │   ├── models.py         # EVA-02 + BLIP model registry
│   │   ├── search.py         # ES kNN search engine
│   │   ├── retrieval.py      # Main retrieval facade
│   │   └── embeddings.py     # Embedding encoder
│   └── pipeline/
│       ├── ingest.py         # Unified IngestPipeline (NEW)
│       ├── indexer.py        # Elasticsearch indexer
│       ├── keyframe.py       # Keyframe extraction
│       ├── embedding.py      # EVA-02 batch embedding
│       ├── detection.py      # Object detection
│       └── ocr.py            # OCR extraction
├── plans/                    # Implementation plans
│   ├── 01_elasticsearch_migration.md
│   └── 02_ingest_pipeline.md
├── docker/
│   └── docker-compose.yml    # Elasticsearch + Kibana
├── data/                     # Data directory (gitignored)
│   ├── keyframes/            # Extracted keyframes
│   ├── embeddings/           # .npy embedding files
│   └── uploads/              # Uploaded files via web UI
└── pyproject.toml
```

### GPU Requirements

| Model | VRAM | Notes |
|---|---|---|
| EVA-02 CLIP Large | ~900 MB | float16 |
| BLIP-ITM Base | ~300 MB | float16 |
| Faster R-CNN | ~200 MB | optional |
| **Total** | **~1.4 GB** | Fits RTX 3050 4GB |

---

## Cấu Hình

Copy `.env.example` thành `.env`:

```env
# Elasticsearch
ES_HOST=localhost
ES_PORT=9200
ES_INDEX_NAME=aic26_keyframes

# Dual embedding weights
EVA02_SEARCH_WEIGHT=0.6
BLIP2_SEARCH_WEIGHT=0.4
USE_DUAL_EMBEDDING=true

# Paths
DATA_DIR=data
KEYFRAMES_DIR=data/keyframes
EMBEDDING_DIR=data/embeddings

# Server
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
```

---

## Đóng Góp

1. Fork repository
2. Tạo branch: `git checkout -b feature/your-feature`
3. Tạo plan: `plans/NN_feature_name.md`
4. Commit: `git commit -m "feat: description"`
5. Push: `git push origin feature/your-feature`
6. Tạo Pull Request

---

## License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">
Built with ❤️ for AI Challenge 2026 · EVA-02 + BLIP-ITM + Elasticsearch
</div>
