"""
Unit tests for core engine modules.
"""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


from aic26.config import settings
from aic26.core.embeddings import EmbeddingIndex
from aic26.core.export import Exporter
from aic26.core.rerank import Reranker


class TestExporter:
    """Tests for Exporter class."""
    
    def test_export_results_empty(self, test_settings):
        """Test that empty results raise ValueError."""
        exporter = Exporter()
        with pytest.raises(ValueError, match="No results"):
            exporter.export_results([], "test")
    
    def test_export_trake_empty(self, test_settings):
        """Test that empty TRAKE payload raises ValueError."""
        exporter = Exporter()
        with pytest.raises(ValueError, match="No TRAKE"):
            exporter.export_trake([], "test")
    
    def test_export_submission_missing_params(self, test_settings):
        """Test that missing parameters raise ValueError."""
        exporter = Exporter()
        with pytest.raises(ValueError, match="required"):
            exporter.export_submission(None, 100, 15)
    
    def test_export_submission_invalid_gap(self, test_settings):
        """Test that invalid gap raises ValueError."""
        exporter = Exporter()
        with pytest.raises(ValueError, match="gap must be > 0"):
            exporter.export_submission("vid", 100, 0)
    
    def test_export_submission_creates_20_rows(self, test_settings, temp_dir):
        """Test submission CSV has exactly 20 rows."""
        settings.output_dir = temp_dir / "submissions"
        settings.output_dir.mkdir(exist_ok=True)
        
        exporter = Exporter()
        path = exporter.export_submission(
            video_name="test_video",
            frame_id=1000,
            gap=15,
            name="test_sub",
        )
        
        assert Path(path).exists()
        
        with open(path, "r") as f:
            rows = f.readlines()
        
        assert len(rows) == 20

    def test_export_submission_with_const_value_quoted(self, test_settings, temp_dir):
        """Test submission CSV with const_value encloses const_value in double quotes."""
        settings.output_dir = temp_dir / "submissions"
        settings.output_dir.mkdir(exist_ok=True)
        
        exporter = Exporter()
        path = exporter.export_submission(
            video_name="L1_V222",
            frame_id=2345,
            gap=15,
            const_value="Truong định",
            name="test_sub_const",
        )
        
        assert Path(path).exists()
        
        with open(path, "r", encoding="utf-8") as f:
            rows = f.readlines()
        
        assert len(rows) == 20
        assert 'L1_V222, 2345, "Truong định"\n' in rows[0]


class TestReranker:
    """Tests for Reranker class."""
    
    def test_reranker_initialization(self):
        """Test that reranker initializes correctly."""
        mock_models = Mock()
        reranker = Reranker(mock_models)
        assert reranker.models is mock_models
    
    def test_reranker_handles_image_open_error(self):
        """Test that reranker handles image loading errors gracefully (keeps candidate with original score)."""
        mock_models = Mock()
        reranker = Reranker(mock_models)
        
        candidates = [
            {"image_path": "/nonexistent/image.jpg", "similarity": 0.9}
        ]
        
        result = reranker.rerank("test query", candidates)
        assert len(result) == 1
        assert result[0]["similarity"] == 0.9
    
    @patch("PIL.Image.open")
    def test_reranker_handles_inference_error(self, mock_open):
        """Test that reranker handles inference errors gracefully."""
        mock_models = Mock()
        mock_models.blip_processor.side_effect = RuntimeError("Inference failed")
        mock_models.device = "cpu"
        
        reranker = Reranker(mock_models)
        candidates = [
            {"image_path": "/fake/image.jpg", "similarity": 0.9}
        ]
        
        result = reranker.rerank("test query", candidates)
        assert result == []

class TestEmbeddingIndex:
    """Tests for EmbeddingIndex."""
    
    def test_cosine_similarity_shape(self):
        """Test cosine similarity returns correct shape."""
        a = np.random.randn(1, 768).astype(np.float32)
        a = a / np.linalg.norm(a, axis=-1, keepdims=True)
        b = np.random.randn(100, 768).astype(np.float32)
        b = b / np.linalg.norm(b, axis=-1, keepdims=True)
        
        sims = EmbeddingIndex.cosine_similarity(a, b)
        assert sims.shape == (1, 100)
    
    def test_cosine_similarity_self_is_one(self):
        """Test that cosine similarity with self is 1.0."""
        a = np.random.randn(10, 768).astype(np.float32)
        a = a / np.linalg.norm(a, axis=-1, keepdims=True)
        sims = EmbeddingIndex.cosine_similarity(a, a)
        
        # Diagonal should be ~1.0
        diag = np.diag(sims)
        np.testing.assert_allclose(diag, 1.0, atol=1e-5)
    
    def test_topk_indices(self):
        """Test top-k index selection."""
        
        
        index = EmbeddingIndex.__new__(EmbeddingIndex)
        sims = np.array([0.1, 0.9, 0.3, 0.7, 0.5], dtype=np.float32)
        
        idx, scores = index.topk_indices(sims, 3)
        
        assert len(idx) == 3
        assert idx[0] == 1  # Highest sim is at index 1
        assert scores[0] == 0.9
    
    def test_threshold_indices(self):
        """Test threshold-based index selection."""
       
        
        index = EmbeddingIndex.__new__(EmbeddingIndex)
        sims = np.array([0.1, 0.9, 0.3, 0.7, 0.5], dtype=np.float32)
        
        idx, scores = index.threshold_indices(sims, 0.5)
        
        assert len(idx) == 3  # 0.9, 0.7, 0.5
        assert idx[0] == 1  # Highest first
    
    def test_threshold_indices_empty(self):
        """Test threshold with no matches returns empty arrays."""
    
        
        index = EmbeddingIndex.__new__(EmbeddingIndex)
        sims = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        idx, scores = index.threshold_indices(sims, 0.9)
        
        assert len(idx) == 0
        assert len(scores) == 0