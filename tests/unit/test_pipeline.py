"""
Unit tests for pipeline modules.
"""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from aic26.src.aic26.pipeline.indexer import IndexBuilder
from aic26.src.aic26.pipeline.keyframe import KeyframeExtractor




class TestKeyframeExtractor:
    """Tests for KeyframeExtractor."""
    
    def test_initialization(self, test_settings):
        """Test extractor initializes correctly."""
        extractor = KeyframeExtractor(test_settings)
        assert extractor.config is test_settings
        assert extractor.device is not None
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors is 1.0."""
        vec = np.array([1.0, 2.0, 3.0])
        sim = KeyframeExtractor._calculate_cosine(vec, vec)
        assert abs(sim - 1.0) < 1e-6
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors is 0.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        sim = KeyframeExtractor._calculate_cosine(vec1, vec2)
        assert abs(sim - 0.0) < 1e-6
    
    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector returns 0.0."""
        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([1.0, 2.0, 3.0])
        sim = KeyframeExtractor._calculate_cosine(vec1, vec2)
        assert sim == 0.0
    
    def test_process_empty_directory(self, test_settings, temp_dir):
        """Test processing empty directory."""
        empty_dir = temp_dir / "empty_videos"
        empty_dir.mkdir()
        
        extractor = KeyframeExtractor(test_settings)
        stats = extractor.process(str(empty_dir))
        
        assert stats["videos_found"] == 0


class TestIndexBuilder:
    """Tests for IndexBuilder."""
    
    def test_initialization(self, test_settings):
        """Test index builder initializes correctly."""
        builder = IndexBuilder(test_settings)
        assert builder.config is test_settings
    
    def test_load_embeddings_empty(self, test_settings, temp_dir):
        """Test loading from empty directory."""
        empty_dir = temp_dir / "empty_embs"
        empty_dir.mkdir()
        
        builder = IndexBuilder(test_settings)
        embs = builder._load_embeddings(str(empty_dir))
        
        assert len(embs) == 0
    
    def test_load_embeddings_with_files(self, test_settings, temp_dir, sample_embeddings):
        """Test loading embeddings from .npy files."""
        emb_dir = temp_dir / "embeddings"
        emb_dir.mkdir()
        
        # Save sample embeddings
        np.save(str(emb_dir / "video1.npy"), sample_embeddings[:50])
        np.save(str(emb_dir / "video2.npy"), sample_embeddings[50:])
        
        builder = IndexBuilder(test_settings)
        embs = builder._load_embeddings(str(emb_dir))
        
        assert len(embs) == 100
        assert embs.shape == (100, 768)
    
    def test_build_index(self, test_settings, temp_dir, sample_embeddings):
        """Test building FAISS index."""
        emb_dir = temp_dir / "embeddings"
        emb_dir.mkdir()
        np.save(str(emb_dir / "video1.npy"), sample_embeddings)
        
        output_path = temp_dir / "test_db.pkl"
        
        builder = IndexBuilder(test_settings)
        stats = builder.build(
            embedding_dir=str(emb_dir),
            output_path=str(output_path),
        )
        
        assert stats["embeddings_loaded"] == 100
        assert stats["dimension"] == 768
        assert Path(output_path).exists()
    
    def test_build_index_directory_not_found(self, test_settings):
        """Test building with non-existent directory."""
        builder = IndexBuilder(test_settings)
        
        with pytest.raises(FileNotFoundError):
            builder.build("/nonexistent/path")