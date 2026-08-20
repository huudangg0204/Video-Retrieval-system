"""
Vietnamese -> English query translation for AIC-26.

Two backends:
- GeminiTranslator: calls the Google Gemini API (used during the real
  competition for maximum translation accuracy).
- LocalTranslator: runs a local Vi->En seq2seq model
  (VietAI/envit5-translation) for cost-free translation during
  development/testing.

Mode "real" retries the Gemini call a configurable number of times with
exponential backoff, then automatically falls back to the local model.
Mode "test" always uses the local model (no network calls, no cost).

A lightweight on-disk JSON cache avoids re-translating repeated queries,
which saves both API cost and latency.

Usage:
    engine = get_translation_engine()
    engine.set_mode("real")            # or "test" (default)
    en_text = engine.translate("Tìm video về ...")
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from aic26.config import settings


# ── Vietnamese detection ─────────────────────────────────────────────────

_VN_CHARS = re.compile(
    "[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
    "ùúụủũưừứựửữỳýỵỷỹđ"
    "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
    "ÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
)


def looks_vietnamese(text: str) -> bool:
    """
    Heuristic check: does the text contain Vietnamese diacritics?

    Used to skip translation for queries that are already in English,
    avoiding unnecessary API calls / local inference.
    """
    return bool(_VN_CHARS.search(text))


# ── Cache ─────────────────────────────────────────────────────────────────

class _TranslationCache:
    """Simple JSON-backed cache: {original_text: translated_text}."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.debug(f"Translation cache loaded: {len(self._data)} entries from {self.path}")
            except Exception as e:
                logger.warning(f"Failed to load translation cache ({e}). Starting empty.")
                self._data = {}
        else:
            logger.debug(f"No translation cache found at {self.path}. Starting empty.")

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._data[key] = value
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to persist translation cache: {e}")


# ── Backends ──────────────────────────────────────────────────────────────

class GeminiTranslator:
    """Translate Vietnamese -> English using the Google Gemini API."""

    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini client initialized (model={self.model_name})")
        return self._client

    def translate(self, text: str) -> str:
        """Translate text via Gemini. Raises on any failure (caller retries)."""
        client = self._get_client()
        prompt = (
            "Translate the following Vietnamese text to English for a video "
            "retrieval search query. Keep proper nouns (place names, person "
            "names, organization names) accurate. Output ONLY the translated "
            "text, with no explanation, quotes, or preamble.\n\n"
            f"{text}"
        )
        response = client.generate_content(prompt)
        translated = (getattr(response, "text", "") or "").strip()
        if not translated:
            raise ValueError("Gemini returned an empty translation")
        return translated


class LocalTranslator:
    """
    Translate Vietnamese -> English using a local seq2seq model.

    Default: VietAI/envit5-translation (T5-based, fine-tuned specifically
    for Vietnamese<->English on the MTet multi-domain corpus). Expects
    inputs prefixed with "vi: " and produces outputs prefixed with "en: ".
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerFast

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading local translation model: {self.model_name} on {self._device}...")
        try:
            self._tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model_name)
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            .to(self._device)
            .eval()
        )
        logger.info("Local translation model ready.")

    def translate(self, text: str) -> str:
        self._load()
        import torch

        prompt = f"vi: {text}"
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_length=512)

        raw = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # envit5 prefixes its output with "en: "
        if raw.lower().startswith("en:"):
            return raw[3:].strip()
        return raw.strip()


# ── Engine ────────────────────────────────────────────────────────────────

class TranslationEngine:
    """
    Vi->En translation with switchable backends and automatic fallback.

    Mode "off":  Disabled / frozen. No translation models loaded (saves VRAM/RAM),
                 no API calls. Text is passed through unchanged.
    Mode "real": Gemini API, retries settings.gemini_max_retries times with
                 exponential backoff, then falls back to the local model.
    Mode "test": Local model only (no network calls, no API cost).

    Usage:
        engine = TranslationEngine()
        engine.set_mode("off")
        text = engine.translate("một truy vấn tiếng việt")  # returns unchanged immediately
    """

    def __init__(self, config=None) -> None:
        self.config = config or settings
        self._mode = (self.config.translation_default_mode or "off").lower().strip()
        if self._mode not in ("off", "test", "real"):
            logger.warning(
                f"Unknown translation mode {self._mode!r}, defaulting to 'off'"
            )
            self._mode = "off"
        self._cache = _TranslationCache(Path(self.config.translation_cache_path))

        self._gemini: Optional[GeminiTranslator] = None
        self._local: LocalTranslator = LocalTranslator(
            self.config.local_translation_model_name
        )
        logger.info(f"TranslationEngine initialized in '{self._mode}' mode")

    # ── Mode control ────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        """Switch between 'off' (disabled), 'test' (local model), and 'real' (Gemini API)."""
        mode = (mode or "").lower().strip()
        if mode not in ("off", "test", "real"):
            raise ValueError(f"Invalid translation mode: {mode!r} (expected 'off', 'test', or 'real')")
        previous = self._mode
        self._mode = mode
        logger.info(f"Translation mode switched: '{previous}' → '{mode}'")

    def get_mode(self) -> str:
        return self._mode

    @property
    def is_enabled(self) -> bool:
        """Return True if translation is active ('test' or 'real')."""
        return self._mode != "off"

    # ── Public API ──────────────────────────────────────────────────

    def translate(self, text: str, force: bool = False) -> str:
        """
        Translate Vietnamese text to English.

        Args:
            text: Input text (may be Vietnamese or already English)
            force: If True, translate even if text doesn't look Vietnamese

        Returns:
            English text (or the original text if mode is 'off', text is
            already English, or force=False)
        """
        if not text or not text.strip():
            return text

        if self._mode == "off":
            logger.debug(f"Translation disabled (mode='off') — skipping: '{text[:50]}...'")
            return text

        if not force and not looks_vietnamese(text):
            logger.debug(f"Skip translation (no Vietnamese detected): '{text[:50]}...'")
            return text

        cached = self._cache.get(text)
        if cached is not None:
            logger.debug(f"Translation cache hit: '{text[:50]}...'")
            return cached

        if self._mode == "real":
            result = self._translate_real(text)
        else:
            result = self._translate_local(text)

        self._cache.set(text, result)
        return result

    # ── Private ─────────────────────────────────────────────────────

    def _translate_real(self, text: str) -> str:
        if self._gemini is None:
            if not self.config.gemini_api_key:
                logger.warning(
                    "Translation mode is 'real' but no Gemini API key is configured "
                    "(GEMINI_API_KEY) — falling back to local translation."
                )
                return self._translate_local(text)
            self._gemini = GeminiTranslator(
                self.config.gemini_api_key, self.config.gemini_model_name
            )

        max_retries = self.config.gemini_max_retries
        backoff = self.config.gemini_retry_backoff_seconds

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Gemini translate attempt {attempt}/{max_retries}: '{text[:60]}...'")
                result = self._gemini.translate(text)
                logger.info(f"Gemini translation OK: '{text[:40]}...' -> '{result[:40]}...'")
                return result
            except Exception as e:
                logger.warning(f"Gemini translate attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    sleep_s = backoff * (2 ** (attempt - 1))
                    logger.debug(f"Retrying in {sleep_s:.1f}s...")
                    time.sleep(sleep_s)

        logger.error(
            f"Gemini translation failed after {max_retries} attempts for "
            f"'{text[:60]}...' — falling back to local model."
        )
        return self._translate_local(text)

    def _translate_local(self, text: str) -> str:
        try:
            logger.info(f"Local translate: '{text[:60]}...'")
            result = self._local.translate(text)
            logger.info(f"Local translation OK: '{text[:40]}...' -> '{result[:40]}...'")
            return result
        except Exception as e:
            logger.error(f"Local translation failed for '{text[:60]}...': {e}")
            return text  # last resort: proceed with the untranslated text


# ── Module-level singleton ─────────────────────────────────────────────────

_engine: Optional[TranslationEngine] = None
_engine_lock = threading.Lock()


def get_translation_engine() -> TranslationEngine:
    """Return the singleton TranslationEngine, creating it on first use."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = TranslationEngine()
    return _engine
