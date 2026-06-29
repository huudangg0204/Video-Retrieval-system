"""
Unit tests for pipeline/captioning.py.

Tests cover:
1. CaptionGenerator — lazy-loads on first .generate(), returns decoded
   text, returns "" on failure, branches on BLIP vs BLIP-2 HF classes.
2. build_caption_fn() — returns a closure that delegates to a
   CaptionGenerator and uses the BLIP base default prompt.

The real BLIP / BLIP-2 models are NOT loaded in tests. We mock at the
``transformers`` boundary (``BlipForConditionalGeneration``,
``BlipProcessor``, ``Blip2ForConditionalGeneration``, ``Blip2Processor``)
so unit tests run quickly without GPU or network. Mirrors the no-GPU
pattern used in tests/unit/test_scene_graph.py.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from aic26.pipeline.captioning import CaptionGenerator, build_caption_fn


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def dummy_image() -> Image.Image:
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


def _stub_blip_classes(monkeypatch, decoded_text: str = "a black cat on a mat"):
    """Patch transformers.*ForConditionalGeneration + *Processor with mocks.

    Returns a tuple ``(model_mock, processor_mock)`` so individual tests
    can inspect call_args.
    """
    model_mock = MagicMock()
    model_mock.device = "cpu"
    processor_mock = MagicMock()
    processor_mock.decode.return_value = decoded_text

    # Wrap from_pretrained in MagicMocks so tests can assert call_count /
    # .called on them.
    blip_processor_cls = MagicMock()
    blip_processor_cls.from_pretrained.return_value = processor_mock
    blip_for_cls = MagicMock()
    # Mirror BlipForConditionalGeneration.from_pretrained(name).to(device)
    chainable = MagicMock()
    chainable.to.return_value = model_mock
    blip_for_cls.from_pretrained.return_value = chainable

    blip2_processor_cls = MagicMock()
    blip2_processor_cls.from_pretrained.return_value = processor_mock
    blip2_for_cls = MagicMock()
    blip2_for_cls.from_pretrained.return_value = model_mock

    import transformers  # local import — only used at patch time

    monkeypatch.setattr(transformers, "BlipProcessor", blip_processor_cls)
    monkeypatch.setattr(transformers, "BlipForConditionalGeneration", blip_for_cls)
    monkeypatch.setattr(transformers, "Blip2Processor", blip2_processor_cls)
    monkeypatch.setattr(transformers, "Blip2ForConditionalGeneration", blip2_for_cls)

    return {
        "model": model_mock,
        "processor": processor_mock,
        "blip_processor_cls": blip_processor_cls,
        "blip_for_cls": blip_for_cls,
        "blip2_processor_cls": blip2_processor_cls,
        "blip2_for_cls": blip2_for_cls,
    }


# ── CaptionGenerator ───────────────────────────────────────────────────


class TestCaptionGenerator:
    """CaptionGenerator lazy-loads and runs generation without crashing."""

    def test_lazy_loads_on_first_generate(self, monkeypatch, dummy_image):
        mocks = _stub_blip_classes(monkeypatch)
        gen = CaptionGenerator(
            model_name="Salesforce/blip-image-captioning-base",
            device="cpu",
        )
        # No load yet.
        assert gen._model is None
        assert gen._processor is None

        gen.generate(dummy_image)

        # After first call, model + processor are set.
        assert gen._model is not None
        assert gen._processor is not None
        # And the model was actually invoked.
        assert mocks["model"].generate.called

    def test_returns_decoded_text(self, monkeypatch, dummy_image):
        _stub_blip_classes(monkeypatch, decoded_text="a black cat on a mat")
        gen = CaptionGenerator(model_name="Salesforce/blip-image-captioning-base", device="cpu")
        out = gen.generate(dummy_image)
        assert out == "a black cat on a mat"

    def test_strips_whitespace(self, monkeypatch, dummy_image):
        _stub_blip_classes(monkeypatch, decoded_text="   a sunset over the hills   ")
        gen = CaptionGenerator(model_name="Salesforce/blip-image-captioning-base", device="cpu")
        out = gen.generate(dummy_image)
        assert out == "a sunset over the hills"  # leading/trailing stripped

    def test_returns_empty_string_on_failure(self, monkeypatch, dummy_image):
        # Make the model raise on .generate().
        mocks = _stub_blip_classes(monkeypatch)
        mocks["model"].generate.side_effect = RuntimeError("CUDA OOM")

        gen = CaptionGenerator(model_name="Salesforce/blip-image-captioning-base", device="cpu")
        out = gen.generate(dummy_image)
        # Graceful degradation: returns "" instead of raising.
        assert out == ""

    def test_returns_empty_string_when_processor_decode_fails(
        self, monkeypatch, dummy_image
    ):
        mocks = _stub_blip_classes(monkeypatch)
        mocks["processor"].decode.side_effect = ValueError("bad token ids")
        gen = CaptionGenerator(model_name="Salesforce/blip-image-captioning-base", device="cpu")
        out = gen.generate(dummy_image)
        assert out == ""

    def test_idempotent_load(self, monkeypatch, dummy_image):
        """Calling generate() twice only invokes from_pretrained once."""
        mocks = _stub_blip_classes(monkeypatch)
        gen = CaptionGenerator(model_name="Salesforce/blip-image-captioning-base", device="cpu")
        gen.generate(dummy_image)
        gen.generate(dummy_image)
        # BlipProcessor.from_pretrained called once during _load, then cached.
        assert mocks["blip_processor_cls"].from_pretrained.call_count == 1

    def test_blip2_branch_uses_blip2_classes(self, monkeypatch, dummy_image):
        """For 'Salesforce/blip2-opt-2.7b', must call Blip2* classes (not Blip*)."""
        mocks = _stub_blip_classes(monkeypatch, decoded_text="a city street at night")
        gen = CaptionGenerator(
            model_name="Salesforce/blip2-opt-2.7b",
            device="cpu",
            quantization="none",
        )
        gen.generate(dummy_image)

        # BLIP-2 classes were used; BLIP base classes were not.
        assert mocks["blip2_processor_cls"].from_pretrained.called
        assert mocks["blip2_for_cls"].from_pretrained.called
        assert not mocks["blip_processor_cls"].from_pretrained.called
        assert not mocks["blip_for_cls"].from_pretrained.called

    def test_close_releases_loaded_state(self, monkeypatch, dummy_image):
        _stub_blip_classes(monkeypatch)
        gen = CaptionGenerator(model_name="Salesforce/blip-image-captioning-base", device="cpu")
        gen.generate(dummy_image)
        assert gen._model is not None
        gen.close()
        assert gen._model is None
        assert gen._processor is None

    def test_close_is_safe_when_never_loaded(self, monkeypatch):
        gen = CaptionGenerator(model_name="Salesforce/blip-image-captioning-base", device="cpu")
        # Should not raise even though _load was never called.
        gen.close()
        assert gen._model is None


# ── build_caption_fn ───────────────────────────────────────────────────


class TestBuildCaptionFn:
    """build_caption_fn() returns a closure that delegates to a CaptionGenerator."""

    def test_returns_callable(self, monkeypatch):
        _stub_blip_classes(monkeypatch, decoded_text="a flower")
        fn = build_caption_fn(
            model_name="Salesforce/blip-image-captioning-base",
            device="cpu",
        )
        assert callable(fn)

    def test_closure_invokes_generator(self, monkeypatch, dummy_image):
        _stub_blip_classes(monkeypatch, decoded_text="a flower")
        fn = build_caption_fn(
            model_name="Salesforce/blip-image-captioning-base",
            device="cpu",
        )
        out = fn(dummy_image)
        assert out == "a flower"

    def test_passes_caption_prompt_for_blip_base(self, monkeypatch, dummy_image):
        """For BLIP base, the build_caption_fn wires in CAPTION_PROMPT as the
        default. BLIP base ignores the prompt, but the wiring must not crash.
        """
        _stub_blip_classes(monkeypatch, decoded_text="x")
        fn = build_caption_fn(
            model_name="Salesforce/blip-image-captioning-base",
            device="cpu",
        )
        # Just verify it runs end-to-end; the prompt is accepted by the
        # BLIP processor (which our mock ignores).
        out = fn(dummy_image)
        assert isinstance(out, str)