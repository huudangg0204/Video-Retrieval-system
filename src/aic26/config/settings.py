"""
Centralized settings using Pydantic Settings.

All configuration is loaded from environment variables with sensible defaults.
Type-safe and validated at startup.

Usage:
    from aic26.config import settings
    
    # Access settings
    print(settings.embedding_dir)
    print(settings.default_top_k)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    
    All paths are relative to the project root unless specified as absolute.
    Environment variables take precedence over defaults.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ── Base paths ──────────────────────────────────────────────────────
    base_dir: Path = Field(
        default_factory=lambda: Path.cwd(),
        description="Base directory of the project",
    )
    
    data_dir: Path = Field(
        default_factory=lambda: Path("data"),
        description="Root data directory",
    )
    
    # ── Data paths ──────────────────────────────────────────────────────
    embedding_dir: Path = Field(
        default_factory=lambda: Path("data/embeddings"),
        description="Directory containing .npy embedding files",
    )
    
    keyframes_dir: Path = Field(
        default_factory=lambda: Path("data/keyframes"),
        description="Directory containing extracted keyframes",
    )
    
    db_path: Path = Field(
        default_factory=lambda: Path("data/faiss_db.pkl"),
        description=(
            "Path to the VectorDatabase query-history pickle file "
            "(core/database.py). NOT the FAISS corpus index — see "
            "faiss_index_path. Kept separate so pipeline/indexer.py "
            "rebuilding the corpus index can never overwrite query history "
            "(or vice versa)."
        ),
    )

    faiss_index_path: Path = Field(
        default_factory=lambda: Path("data/faiss_index.pkl"),
        description=(
            "Path to the FAISS corpus index pickle built by "
            "pipeline/indexer.py::IndexBuilder (build() / build_dual()). "
            "Separate from db_path — previously both defaulted to the same "
            "file, which meant rebuilding the index silently clobbered "
            "VectorDatabase's query history and vice versa."
        ),
    )
    
    output_dir: Path = Field(
        default_factory=lambda: Path("data/submissions"),
        description="Directory for CSV submission exports",
    )
    
    map_keyframe_dir: Path = Field(
        default_factory=lambda: Path("data/map-keyframes"),
        description="Directory for keyframe mapping CSV files",
    )
    
    media_info_dir: Path = Field(
        default_factory=lambda: Path("data/media-info"),
        description="Directory for video media info JSON files",
    )
    
    upload_dir: Path = Field(
        default_factory=lambda: Path("data/uploads"),
        description="Directory for uploaded images",
    )
    
    detection_dir: Path = Field(
        default_factory=lambda: Path("data/detections"),
        description="Directory for object detection results",
    )
    
    ocr_output_path: Path = Field(
        default_factory=lambda: Path("data/ocr_results.csv"),
        description="Path for OCR output CSV",
    )
    
    # ── Model configuration ─────────────────────────────────────────────
    device: str = Field(
        default="cuda",
        description="Device to use for models (cpu, cuda, cuda:0, etc.)",
    )
    
    eva02_model_name: str = Field(
        default="hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k",
        description="EVA-02 CLIP model identifier from HuggingFace/TIMM",
    )
    
    eva02_pretrained: str = Field(
        default="laion2b_s4b_b131k",
        description="Pretrained weights identifier for EVA-02",
    )
    
    blip_model_name: str = Field(
        default="Salesforce/blip-itm-base-coco",
        description="BLIP Image-Text Matching model identifier",
    )

    clip32_model_name: str = Field(
        default="openai/clip-vit-base-patch32",
        description="CLIP ViT-B-32 model identifier (HuggingFace) for dual-embedding pipeline",
    )

    clip32_embedding_dir: Path = Field(
        default_factory=lambda: Path("data/clip32_embeddings"),
        description="Directory containing .npy CLIP ViT-B-32 embedding files",
    )

    use_dual_embedding: bool = Field(
        default=True,
        description="If true, encode with EVA-02 + BLIP-2 and weighted-blend cosine similarity at search time",
    )

    eva02_search_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight of EVA-02 cosine similarity in the dual-embedding blend",
    )

    clip32_search_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight of CLIP ViT-B-32 cosine similarity in the dual-embedding blend",
    )

    dual_search_pool_multiplier: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Multiplier applied to top_k when collecting candidates from each stream before blending",
    )

    # ── BM25 Caption Stream ────────────────────────────────────────
    bm25_search_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight of BM25 caption similarity when blending with EVA-02/CLIP-32 streams",
    )

    yolo_model_name: str = Field(
        default="yolo26x.pt",
        description="YOLOv26 model filename or path (ultralytics format)",
    )

    oid_mapping_csv: str = Field(
        default="oidv7-class-descriptions-boxable.csv",
        description="Open Images V7 class descriptions CSV path",
    )

    # ── Metadata extraction (per docs/specs/metadata-extraction.md) ──────
    metadata_dir: Path = Field(
        default_factory=lambda: Path("data/metadata"),
        description="Directory for per-keyframe metadata JSON sidecars (detections + scene + scene_graph)",
    )

    scene_graph_model: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        description="MLLM model for scene-graph (subject, predicate, object) extraction",
    )

    scene_graph_max_triples: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of (subject, predicate, object) triples per image (SnapSeek §3.5)",
    )

    scene_graph_max_new_tokens: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Max new tokens for scene-graph JSON generation",
    )

    scene_graph_quantization: str = Field(
        default="4bit",
        description="Quantization for scene-graph MLLM: '4bit' (bitsandbytes), '8bit', or 'none'",
    )

    places365_model_name: str = Field(
        default="efficientnet_b2_places365",
        description="Places365 model identifier (lifeXplore §3 Table 1)",
    )

    places365_weights_path: Optional[Path] = Field(
        default=None,
        description="Optional path to Places365 weights file; auto-resolved if None",
    )

    position_min_bbox_fraction: float = Field(
        default=0.05,
        gt=0.0,
        le=0.5,
        description="Minimum bbox area as fraction of image to emit a position label (else None)",
    )

    extract_scene_graph: bool = Field(
        default=True,
        description="Run Qwen2.5-VL scene graph during extract-metadata (set false to skip slow step)",
    )

    ocr_csv_path: Path = Field(
        default_factory=lambda: Path("data/ocr_results.csv"),
        description="Path to OCR results CSV produced by pipeline/ocr.py; consumed by 'merge-ocr' subcommand",
    )

    # ── Image captioning (Task 3.2) ────────────────────────────────
    caption_model_name: str = Field(
        default="Salesforce/blip-image-captioning-base",
        description=(
            "Captioning model id. Default BLIP base (446M, ~2GB VRAM, fast). "
            "Switch to 'Salesforce/blip2-opt-2.7b' for richer captions (~5GB)."
        ),
    )

    caption_max_new_tokens: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Max tokens for caption generation.",
    )

    caption_min_length: int = Field(
        default=8,
        ge=1,
        le=200,
        description="Min tokens for caption generation; suppresses single-word output.",
    )

    caption_quantization: str = Field(
        default="none",
        description=(
            "Quantization for the caption model. 'none' for the 446M default. "
            "Set '8bit' / '4bit' (bitsandbytes) if switching to BLIP-2 OPT 2.7b."
        ),
    )

    # ── Vi->En query translation (core/translation.py) ──────────────────
    translation_default_mode: str = Field(
        default="test",
        description=(
            "'test' = local envit5 model only, no network/API cost (safe "
            "default for dev). 'real' = Gemini API with local fallback on "
            "failure — set this to 'real' for the actual competition run "
            "so translation quality is as high as possible."
        ),
    )

    local_translation_model_name: str = Field(
        default="VietAI/envit5-translation",
        description="HF seq2seq model for local Vi->En translation (mode='test' or Gemini fallback)",
    )

    translation_cache_path: Path = Field(
        default_factory=lambda: Path("data/translation_cache.json"),
        description="On-disk JSON cache of {original_text: translated_text}, avoids re-translating repeats",
    )

    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key for mode='real' translation. Required for 'real' mode — "
                    "falls back to the local model with a warning if unset.",
    )

    gemini_model_name: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model id used for Vi->En query translation",
    )

    gemini_max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Retry attempts for a Gemini translation call before falling back to the local model",
    )

    gemini_retry_backoff_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        description="Base seconds for exponential backoff between Gemini retries",
    )

    # ── Search parameters ──────────────────────────────────────────────
    default_top_k: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Default number of search results to return",
    )
    
    default_score_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Default minimum similarity score for threshold searches",
    )
    
    default_obj_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default confidence threshold for object detection filtering",
    )
    
    rerank_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="BLIP ITM probability threshold for reranking",
    )
    
    # ── TRAKE parameters ───────────────────────────────────────────────
    gap_keyframe: int = Field(
        default=15,
        ge=1,
        description="Gap between keyframes (for video ~25-30fps)",
    )
    
    trake_default_top_k: int = Field(
        default=200,
        ge=1,
        description="Default top-k per event for TRAKE",
    )
    
    trake_default_candidates: int = Field(
        default=200,
        ge=1,
        description="Default number of candidate solutions for TRAKE",
    )
    
    # ── Server configuration ───────────────────────────────────────────
    host: str = Field(
        default="0.0.0.0",
        description="Server host address",
    )
    
    port: int = Field(
        default=5000,
        ge=1,
        le=65535,
        description="Server port",
    )
    
    debug: bool = Field(
        default=False,
        description="Enable Flask debug mode",
    )
    
    flask_secret_key: str = Field(
        default="aic26-dev-secret-change-in-production",
        description="Flask secret key for session management",
    )
    
    max_upload_size_mb: int = Field(
        default=16,
        ge=1,
        le=1024,
        description="Maximum upload file size in MB",
    )
    
    # ── Pipeline parameters ────────────────────────────────────────────
    max_workers: int = Field(
        default=2,
        ge=1,
        le=32,
        description="Maximum number of worker threads for parallel processing",
    )
    
    keyframe_interval: float = Field(
        default=1.0,
        gt=0.0,
        description="Interval in seconds between keyframe extraction checks",
    )
    
    keyframe_similarity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for keyframe extraction",
    )
    
    keyframe_auto_accept_seconds: float = Field(
        default=10.0,
        gt=0.0,
        description="Auto-accept keyframe if this many seconds since last accepted",
    )
    
    # ── Logging ────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    
    log_file: Optional[Path] = Field(
        default=None,
        description="Path to log file (None for console only)",
    )
    
    log_format: str = Field(
        default="structured",
        description="Log format: 'structured' or 'simple'",
    )
    
    # ── Validators ─────────────────────────────────────────────────────
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper

    @model_validator(mode="after")
    def resolve_metadata_dir(self) -> "Settings":
        """If metadata_dir is relative and data_dir was overridden, anchor metadata_dir under data_dir.

        Mirrors the behavior expected by tests: passing data_dir=temp_dir/data
        should also relocate metadata_dir to temp_dir/data/metadata when the user
        has not set METADATA_DIR explicitly. Absolute paths are left untouched.
        """
        md = self.metadata_dir
        if not md.is_absolute():
            # Only auto-relocate when default value is in use; if user set METADATA_DIR
            # to a relative path explicitly, honor it relative to cwd.
            default_md = Path("data/metadata")
            if md == default_md:
                self.metadata_dir = self.data_dir / "metadata"
        return self
    
    # ── Utility methods ────────────────────────────────────────────────
    
    def ensure_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.embedding_dir,
            self.keyframes_dir,
            self.output_dir,
            self.map_keyframe_dir,
            self.media_info_dir,
            self.upload_dir,
            self.detection_dir,
            self.clip32_embedding_dir,
            self.metadata_dir,
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def is_gpu_available(self) -> bool:
        """Check if CUDA GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def display(self) -> None:
        """Print all current settings (for debugging)."""
        from loguru import logger
        logger.info("Current Settings:")
        for field_name, field_info in self.model_fields.items():
            value = getattr(self, field_name)
            logger.info(f"  {field_name}: {value}")


# ── Singleton instance ────────────────────────────────────────────────────

settings = Settings()

# Auto-create directories on import
settings.ensure_directories()