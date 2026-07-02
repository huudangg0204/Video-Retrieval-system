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
CLIP32_EMBEDDING_DIM: Final[int] = 512  # openai/clip-vit-base-patch32 projection dim

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

# ── Metadata extraction (per docs/specs/metadata-extraction.md) ─────────
# Bbox position grid (3×3): row (top/center/bottom) × col (left/center/right).
# Used by pipeline/position.py to classify each YOLO detection.
POSITION_GRID_3x3: Final[tuple[str, ...]] = (
    "top-left", "top", "top-right",
    "left", "center", "right",
    "bottom-left", "bottom", "bottom-right",
)
# Minimum bbox area as fraction of image area to emit a position label.
# Below this, position is unreliable (object too small) and returns None.
POSITION_GRID_THRESHOLD: Final[float] = 0.05

# Scene graph prompt template (SnapSeek 3.0 §3.5, LSC'25).
# Rules: lowercase noun, base-form predicates, no passive constructions,
# JSON-only output, top-K most salient triples.
SCENE_GRAPH_PROMPT: Final[str] = """Analyze this image and list all visible subject-predicate-object triples.

Rules:
- Use lowercase nouns for subjects and objects (e.g., 'person', 'cup', 'table')
- Use base-form verbs for predicates (e.g., 'hold', 'sit on', 'next to', 'in front of')
- Avoid passive constructions
- Only describe triples you can clearly see
- Limit to the top 5 most salient triples

Output format (JSON only, no commentary):
{"triples": [["subject", "predicate", "object"], ...]}"""

# Default Places365 model identifier (lifeXplore §3 Table 1 — EfficientNet-B2).
PLACES365_DEFAULT_MODEL: Final[str] = "efficientnet_b2_places365"

# Scene graph 3-tuple max triples per image (SnapSeek 3.0 §3.5 "top 5").
SCENE_GRAPH_MAX_TRIPLES: Final[int] = 5

# Default scene-graph MLLM model identifier (SnapSeek 3.0 §3.5).
SCENE_GRAPH_DEFAULT_MODEL: Final[str] = "Qwen/Qwen2.5-VL-7B-Instruct"

# Default max new tokens for scene-graph JSON generation.
SCENE_GRAPH_MAX_NEW_TOKENS: Final[int] = 512

# ── Image captioning (Task 3.2) ─────────────────────────────────────

# Default captioning model. BLIP base is intentionally separate from
# settings.blip2_model_name (blip2-itm-vit-g) — that ITM variant
# has no .generate() and cannot caption. See memory
# blip2-itm-vit-g-image-only.md for the full diagnosis.
CAPTION_DEFAULT_MODEL: Final[str] = "Salesforce/blip-image-captioning-base"

# Default prompt prepended to the image for conditional captioning.
# BLIP base was trained with short unconditional prompts; BLIP-2 also
# accepts an empty / domain-specific prompt.
CAPTION_PROMPT: Final[str] = "a photo of"

# Max caption length in tokens. BLIP base COCO captions average ~10
# tokens so 50 is generous; operators can raise to 200 for verbose outputs.
CAPTION_MAX_NEW_TOKENS: Final[int] = 50

# Min caption length — suppresses degenerate single-word outputs ("a", "the").
CAPTION_MIN_LENGTH: Final[int] = 8
