"""
Unit tests for translation toggle (off / test / real) and VRAM optimization.
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from aic26.config import Settings
from aic26.core.translation import TranslationEngine, LocalTranslator, looks_vietnamese


class TestTranslationToggle:
    """Tests for TranslationEngine mode switching and zero-overhead 'off' mode."""

    def test_translation_engine_defaults_to_off_mode(self):
        """Test that default TranslationEngine starts in 'off' mode with no models loaded."""
        config = Settings(translation_default_mode="off")
        engine = TranslationEngine(config=config)

        assert engine.get_mode() == "off"
        assert engine.is_enabled is False
        assert engine._local._model is None
        assert engine._gemini is None

    def test_mode_off_returns_original_text_immediately(self):
        """Test that mode='off' bypasses translation completely and returns input unchanged."""
        config = Settings(translation_default_mode="off")
        engine = TranslationEngine(config=config)

        vi_text = "một chiếc xe ô tô màu đỏ đang chạy trên đường"
        result = engine.translate(vi_text, force=True)

        assert result == vi_text
        # Ensure LocalTranslator was NOT loaded
        assert engine._local._model is None

    def test_set_mode_validation(self):
        """Test switching modes between off, test, and real."""
        engine = TranslationEngine(config=Settings(translation_default_mode="off"))

        engine.set_mode("test")
        assert engine.get_mode() == "test"
        assert engine.is_enabled is True

        engine.set_mode("real")
        assert engine.get_mode() == "real"
        assert engine.is_enabled is True

        engine.set_mode("off")
        assert engine.get_mode() == "off"
        assert engine.is_enabled is False

        with pytest.raises(ValueError, match="Invalid translation mode"):
            engine.set_mode("invalid_mode")

    @patch.object(LocalTranslator, "translate", return_value="a blue submarine in the ocean")
    def test_mode_test_invokes_local_translator(self, mock_local_translate, tmp_path):
        """Test that mode='test' invokes the local translator when given Vietnamese text."""
        cache_file = tmp_path / "temp_cache.json"
        engine = TranslationEngine(config=Settings(translation_default_mode="test", translation_cache_path=cache_file))
        vi_text = "một chiếc tàu ngầm màu xanh dưới đại dương"

        result = engine.translate(vi_text)

        assert result == "a blue submarine in the ocean"
        mock_local_translate.assert_called_once_with(vi_text)

    def test_english_query_skips_translation_even_when_on(self):
        """Test that plain English queries skip translation even if mode='test'."""
        engine = TranslationEngine(config=Settings(translation_default_mode="test"))
        en_text = "a red car driving on the street"

        result = engine.translate(en_text, force=False)
        assert result == en_text
        assert engine._local._model is None
