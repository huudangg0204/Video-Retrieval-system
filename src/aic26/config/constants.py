"""
Non-configurable constants for the AIC-26 project.

These are values that are inherent to the models/datasets used
and should not change based on configuration.
"""
from __future__ import annotations

from typing import Final

# ── COCO Instance Segmentation Categories ───────────────────────────────
# 91 categories (index 0 is background)
COCO_CATEGORIES: Final[list[str]] = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]

# ── File extension filters ──────────────────────────────────────────────
IMAGE_EXTENSIONS: Final[tuple[str, ...]] = (
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'
)

VIDEO_EXTENSIONS: Final[tuple[str, ...]] = (
    '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'
)

# ── Model output dimensions ─────────────────────────────────────────────
EVA02_EMBEDDING_DIM: Final[int] = 768
BLIP_ITM_DIM: Final[int] = 768
DINOV2_EMBEDDING_DIM: Final[int] = 384
BLIP2_EMBEDDING_DIM: Final[int] = 768  # blip2-itm-vit-g Q-Former pooled query output

# ── FAISS index types ───────────────────────────────────────────────────
FAISS_INDEX_TYPES: Final[dict[str, str]] = {
    "flat_l2": "IndexFlatL2",
    "flat_ip": "IndexFlatIP",  # Inner Product (for cosine with normalized vectors)
    "ivf_flat": "IndexIVFFlat",
    "ivf_pq": "IndexIVFPQ",
}

# ── Supported languages for OCR ─────────────────────────────────────────
OCR_LANGUAGES: Final[list[str]] = ['vi', 'en', 'ch', 'japan', 'korean']

# ── TRAKE strategy names ────────────────────────────────────────────────
TRAKE_STRATEGIES: Final[list[str]] = ["closest", "highest"]

# ── API rate limits ─────────────────────────────────────────────────────
API_RATE_LIMIT: Final[int] = 100  # requests per minute
API_BURST_LIMIT: Final[int] = 10  # burst requests

# ── Submission format ───────────────────────────────────────────────────
SUBMISSION_MAX_ROWS: Final[int] = 20
SUBMISSION_DEFAULT_GAP: Final[int] = 15