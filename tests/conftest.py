"""
Shared test fixtures for AIC-26.

Provides common test dependencies:
- Temporary directories
- Sample settings
- Mock embeddings and results
"""
from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
import numpy as np

from aic26.src.aic26.config.settings import Settings




@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory that is cleaned up after each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="function")
def test_settings(temp_dir):
    """Create test settings with temporary directories."""
    data_dir = temp_dir / "data"
    settings = Settings(
        data_dir=data_dir,
        embedding_dir=data_dir / "embeddings",
        keyframes_dir=data_dir / "keyframes",
        db_path=data_dir / "test_db.pkl",
        output_dir=data_dir / "submissions",
        map_keyframe_dir=data_dir / "map-keyframes",
        media_info_dir=data_dir / "media-info",
        upload_dir=data_dir / "uploads",
        detection_dir=data_dir / "detections",
        debug=True,
        log_level="DEBUG",
    )
    settings.ensure_directories()
    return settings


@pytest.fixture(scope="module")
def sample_embeddings():
    """Create sample embedding array (100 frames, 768 dimensions)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 768), dtype=np.float32)


@pytest.fixture(scope="module")
def sample_results():
    """Create sample search results list."""
    return [
        {
            "image_path": f"/path/to/frame_{i:03d}.jpg",
            "video_name": "test_video",
            "similarity": 0.95 - i * 0.05,
            "frame_idx": f"{i:03d}",
        }
        for i in range(10)
    ]


@pytest.fixture(scope="module")
def sample_video_names():
    """Create sample video name list."""
    return [f"L{str(i).zfill(2)}_V{str(j).zfill(3)}" for i in range(1, 4) for j in range(1, 4)]