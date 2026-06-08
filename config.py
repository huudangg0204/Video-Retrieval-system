"""
config.py — Centralized configuration for AIC2025 retrieval system.
All path constants and model settings live here.
Override any value with the corresponding environment variable.
"""
import os

# ── Paths ────────────────────────────────────────────────────────────────────
EMBEDDING_DIR     = os.environ.get("EMBEDDING_DIR",   r"./Batch2-btc/eva02_large_patch14_clip_224.merged2b_s4b_b131k")
KEYFRAMES_DIR     = os.environ.get("KEYFRAMES_DIR",   r"./keyframes")
DB_PATH           = os.environ.get("DB_PATH",         r"faiss_db.pkl")
OUTPUT_FOLDER     = os.environ.get("OUTPUT_FOLDER",   r"./submission")
MAP_KEYFRAME_PATH = os.environ.get("MAP_KEYFRAME_PATH", r"./map-keyframes")
MEDIA_INFO_PATH   = os.environ.get("MEDIA_INFO_PATH",  r"./media-info")
UPLOAD_DIR        = os.environ.get("UPLOAD_DIR",       r"./_uploads")

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME        = "hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k"
MODEL_PRETRAINED  = "laion2b_s4b_b131k"
BLIP_MODEL_NAME   = "Salesforce/blip-itm-base-coco"

# ── Search defaults ───────────────────────────────────────────────────────────
DEFAULT_TOP_K     = 100
DEFAULT_SCORE_THRESHOLD = 0.2
DEFAULT_OBJ_THRESHOLD   = 0.5
RERANK_THRESHOLD        = 0.4

# ── Temporal (TRAKE) ──────────────────────────────────────────────────────────
GAP_KEYFRAME      = 15   # video ~25-30fps; 15 frames safety gap

# ── COCO categories (index 0 = background, valid labels start at 1) ───────────
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]
