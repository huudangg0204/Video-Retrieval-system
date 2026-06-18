# Plan: Unified Data Ingestion Pipeline

> Created: 2026-06-18  
> Status: **COMPLETED**  
> Branch: `feature/elasticsearch-migration`

## Mục Tiêu

Tạo một entry point duy nhất để ingest dữ liệu vào hệ thống retrieval, hỗ trợ 2 dạng input:
- **Mode A**: Keyframe folder/ZIP (Kaggle dataset format)
- **Mode B**: Raw video files

## Input Formats

### Mode A — Kaggle Keyframe Dataset
```
input/
├── L01_V001/
│   ├── 000.jpg
│   └── 001.jpg
└── L01_V002/
    └── ...
```
Cũng chấp nhận file `.zip` với cấu trúc tương tự bên trong.

### Mode B — Video Files
```
videos/
├── L01_V001.mp4
└── L01_V002.mp4
```

## Processing Pipeline (shared)

```
Input
  │
  ├─[Mode B]─ KeyframeExtractor → data/keyframes/
  │
  ▼
[Optional] ObjectDetector     → data/detections/
  │
  ▼
[Optional] OCRExtractor       → data/ocr_results.csv
  │
  ▼
EmbeddingExtractor (EVA-02)   → data/embeddings/*.npy
  │
  ▼
ElasticsearchIndexer (bulk)   → ES index aic26_keyframes
```

## Files Created

- `pipeline/ingest.py` [NEW] — `IngestPipeline` class + `IngestStats` dataclass
- `cli/main.py` — `aic26 ingest` command group (keyframes / videos / auto)

## CLI Usage

```bash
# Mode A: Kaggle keyframe dataset (folder)
py -3.10 -m aic26.cli.main ingest keyframes --input data/keyframes/

# Mode A: ZIP file
py -3.10 -m aic26.cli.main ingest keyframes --input dataset.zip

# Mode B: Video files
py -3.10 -m aic26.cli.main ingest videos --input data/videos/

# Auto-detect
py -3.10 -m aic26.cli.main ingest auto --input data/

# Common options
#   --skip-detection   Bỏ qua object detection
#   --skip-ocr         Bỏ qua OCR
#   --skip-es          Chỉ extract embedding, không index ES
#   --force-recreate   Xóa ES index và tạo lại
#   --no-blip          EVA-02 only (không BLIP)
#   --batch-size N     GPU batch size (default: 40)
```
