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
from pydantic import Field, field_validator
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
        description="Path to the FAISS database pickle file",
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
    
    yolo_model_name: str = Field(
        default="yolov8x-oiv7.pt",
        description="YOLOv8 model filename or path",
    )
    
    oid_mapping_csv: str = Field(
        default="oidv7-class-descriptions-boxable.csv",
        description="Open Images V7 class descriptions CSV path",
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