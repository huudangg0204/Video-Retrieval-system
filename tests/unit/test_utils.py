"""
Unit tests for utility functions.
"""
from __future__ import annotations

import pytest
from pathlib import Path
import pandas as pd
import numpy as np

from aic26.src.aic26.utils.file_utils import ensure_dir, find_files, get_file_stem, safe_basename
from aic26.src.aic26.utils.frame_utils import detect_map_columns, parse_frame_int, parse_frame_number_from_filename
from aic26.src.aic26.utils.media_utils import frame_to_timestr




class TestFileUtils:
    """Tests for file utility functions."""
    
    def test_ensure_dir_creates_directory(self, temp_dir):
        """Test that ensure_dir creates a directory."""
        new_dir = temp_dir / "test" / "nested"
        result = ensure_dir(new_dir)
        
        assert result.exists()
        assert result.is_dir()
    
    def test_ensure_dir_existing_directory(self, temp_dir):
        """Test that ensure_dir works with existing directory."""
        temp_dir.mkdir(exist_ok=True)
        result = ensure_dir(temp_dir)
        
        assert result.exists()
    
    def test_safe_basename_normal(self):
        """Test safe_basename with normal path."""
        result = safe_basename("/path/to/file.txt")
        assert result == "file.txt"
    
    def test_safe_basename_none(self):
        """Test safe_basename with None input."""
        result = safe_basename(None)
        assert result == "None"
    
    def test_safe_basename_empty(self):
        """Test safe_basename with empty string."""
        result = safe_basename("")
        assert result == ""
    
    def test_get_file_stem(self):
        """Test get_file_stem extraction."""
        result = get_file_stem("/path/to/frame_042.jpg")
        assert result == "frame_042"
    
    def test_find_files(self, temp_dir):
        """Test finding files by extension."""
        # Create test files
        (temp_dir / "image1.jpg").touch()
        (temp_dir / "image2.png").touch()
        (temp_dir / "document.txt").touch()
        
        images = find_files(temp_dir, extensions=(".jpg", ".png"))
        
        assert len(images) == 2
        assert any("image1.jpg" in str(p) for p in images)
        assert any("image2.png" in str(p) for p in images)


class TestFrameUtils:
    """Tests for frame utility functions."""
    
    @pytest.mark.parametrize("filename,expected", [
        ("frame_001.jpg", 1),
        ("042.png", 42),
        ("000042.jpeg", 42),
        ("video_frame_123", 123),
        ("image.jpg", None),
        ("no_digits.png", None),
        ("", None),
    ])
    def test_parse_frame_number(self, filename, expected):
        """Test frame number parsing with various formats."""
        result = parse_frame_number_from_filename(filename)
        assert result == expected
    
    def test_detect_map_columns_standard(self):
        """Test column detection with standard column names."""
        df = pd.DataFrame({
            "n": [1, 2, 3],
            "frame_index": [0, 30, 60],
            "fps": [30.0, 30.0, 30.0],
        })
        
        n_col, idx_col = detect_map_columns(df)
        assert n_col == "n"
        assert idx_col == "frame_index"
    
    def test_detect_map_columns_alternative(self):
        """Test column detection with alternative column names."""
        df = pd.DataFrame({
            "frame_number": [1, 2, 3],
            "idx": [0, 30, 60],
        })
        
        n_col, idx_col = detect_map_columns(df)
        assert n_col == "frame_number"
        assert idx_col == "idx"
    
    def test_detect_map_columns_missing(self):
        """Test column detection raises error when columns missing."""
        df = pd.DataFrame({"unknown_col": [1, 2, 3]})
        
        with pytest.raises(KeyError):
            detect_map_columns(df)
    
    @pytest.mark.parametrize("input_val,expected", [
        (42, 42),
        ("42", 42),
        ("042", 42),
        ("frame_042.jpg", 42),
        ("abc", 0),
        (None, 0),
    ])
    def test_parse_frame_int(self, input_val, expected):
        """Test frame int parsing with various inputs."""
        result = parse_frame_int(input_val)
        assert result == expected


class TestMediaUtils:
    """Tests for media utility functions."""
    
    @pytest.mark.parametrize("frame_idx,fps,expected", [
        (0, 30, "0:00"),
        (90, 30, "0:03"),
        (1800, 30, "1:00"),
        (3600, 30, "2:00"),
        (0, 0, "0:00"),
        (100, None, "0:00"),
    ])
    def test_frame_to_timestr(self, frame_idx, fps, expected):
        """Test frame to time string conversion."""
        result = frame_to_timestr(frame_idx, fps)
        assert result == expected