"""
Unit tests for configuration settings.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from aic26.src.aic26.config.settings import Settings




class TestSettings:
    """Tests for Settings class."""
    
    def test_default_settings(self):
        """Test that default settings are created correctly."""
        settings = Settings()
        
        assert settings.default_top_k == 100
        assert settings.host == "0.0.0.0"
        assert settings.port == 5000
        assert settings.log_level == "INFO"
    
    def test_custom_settings(self):
        """Test that custom settings override defaults."""
        settings = Settings(
            default_top_k=50,
            port=8080,
            log_level="DEBUG",
        )
        
        assert settings.default_top_k == 50
        assert settings.port == 8080
        assert settings.log_level == "DEBUG"
    
    def test_log_level_validation(self):
        """Test that invalid log levels are rejected."""
        with pytest.raises(ValueError):
            Settings(log_level="INVALID")
    
    def test_log_level_uppercase(self):
        """Test that log levels are uppercased."""
        settings = Settings(log_level="debug")
        assert settings.log_level == "DEBUG"
    
    def test_ensure_directories(self, temp_dir):
        """Test that directories are created."""
        settings = Settings(data_dir=temp_dir / "data")
        settings.ensure_directories()
        
        assert settings.data_dir.exists()
        assert settings.embedding_dir.exists()
        assert settings.keyframes_dir.exists()
    
    def test_is_gpu_available(self):
        """Test GPU availability check (may be False in CI)."""
        settings = Settings()
        result = settings.is_gpu_available()
        assert isinstance(result, bool)
    
    def test_display(self, capsys):
        """Test that display method doesn't crash."""
        settings = Settings()
        settings.display()
        # Should not raise any exception