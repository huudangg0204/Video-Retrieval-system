"""
Per-image caption generation via BLIP / BLIP-2.

Mirrors the MLLM pattern from pipeline/scene_graph.py: a thin wrapper class
``CaptionGenerator`` lazy-loads the HuggingFace model + processor on first
call and exposes ``generate(image) -> str``. ``build_caption_fn()`` returns
a closure suitable for ``MetadataExtractor.caption_fn``.

Why a separate model from ``core/models.py``
--------------------------------------------
``core/models.py::_load_blip2()`` uses ``Blip2Model`` (pooled Q-Former
output for the dual-embedding search pipeline) and is configured with
``Salesforce/blip2-itm-vit-g`` — the **image-text matching (ITM)** variant.
The ITM head is a classifier over ``(image, text)`` pairs; it has no
``.generate()`` method. Captioning needs ``Blip2ForConditionalGeneration``
(a different HuggingFace class) and a captioning-trained checkpoint
(BLIP base, BLIP-2 OPT, BLIP-2 Flan-T5). See the memory note
``blip2-itm-vit-g-image-only`` for the full diagnosis.

Design notes
------------
- **Lazy load**: the model is loaded on the first call to ``.generate()``,
  not at construction. Lets operators defer VRAM allocation until
  captioning is actually requested. Matches the Qwen scene-graph pattern
  in ``metadata.py:build_metadata_callables``.
- **Graceful degradation**: any failure inside ``.generate()`` logs a
  warning and returns ``""``. The metadata pipeline must never halt over
  a single bad caption — same contract as scene-graph and detection.
- **Branch on model family**: BLIP base and BLIP-2 use different HF
  classes and processor signatures. We pick based on whether
  ``"blip2"`` appears in the model name (case-insensitive).
- **Quantization**: optional 4-bit / 8-bit via ``bitsandbytes`` for the
  larger BLIP-2 checkpoints (e.g. ``blip2-opt-2.7b``). Defaults to
  ``"none"`` which is fine for the 446M BLIP base.

References
----------
- SalesForce BLIP: https://arxiv.org/abs/2201.12086
- BLIP-2 (Q-Former): https://arxiv.org/abs/2301.12597
- docs/specs/metadata-extraction.md §Open Questions (Task 3.2).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PIL import Image

from aic26.config.constants import CAPTION_PROMPT

if TYPE_CHECKING:
    from aic26.pipeline.metadata import CaptionFn

logger = logging.getLogger(__name__)


class CaptionGenerator:
    """Lazy-loaded BLIP / BLIP-2 caption generator.

    Designed to be created once and called many times. The first call to
    :meth:`generate` loads the HuggingFace model + processor onto the
    configured device; subsequent calls reuse them. :meth:`close` releases
    GPU memory.

    Args:
        model_name: HuggingFace model id. Use a captioning-trained
            checkpoint — ``Salesforce/blip-image-captioning-base``
            (446M, default, ~2 GB VRAM) or ``Salesforce/blip2-opt-2.7b``
            (~5 GB VRAM, richer captions).
        device: Torch device string (``"cuda"``, ``"cuda:0"``, ``"cpu"``).
        max_new_tokens: Max tokens for ``.generate()``. Default 50
            matches BLIP base's COCO-caption training length.
        min_length: Min tokens for ``.generate()``. Default 8 suppresses
            degenerate single-word outputs (``"a"``, ``"the"``).
        quantization: ``"none"`` (default), ``"8bit"``, or ``"4bit"``.
            4bit/8bit require ``bitsandbytes`` and target the larger
            BLIP-2 checkpoints.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = "cpu",
        max_new_tokens: int = 50,
        min_length: int = 8,
        quantization: str = "none",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.min_length = min_length
        self.quantization = quantization
        self._model = None
        self._processor = None

    # ── Loading ──────────────────────────────────────────────────

    def _load(self) -> None:
        """Lazy-load the HuggingFace model + processor.

        Branches on whether ``"blip2"`` appears in ``model_name`` so the
        right ``*ForConditionalGeneration`` class + processor are used.
        Idempotent: subsequent calls are no-ops once ``_model`` is set.
        """
        if self._model is not None:
            return

        is_blip2 = "blip2" in self.model_name.lower()
        logger.info(
            "Loading caption model %s on %s (blip2=%s, quant=%s)",
            self.model_name,
            self.device,
            is_blip2,
            self.quantization,
        )

        if is_blip2:
            from transformers import Blip2ForConditionalGeneration, Blip2Processor

            self._processor = Blip2Processor.from_pretrained(self.model_name)
            load_kwargs: dict = {}
            if self.quantization in ("4bit", "8bit"):
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=(self.quantization == "4bit"),
                    load_in_8bit=(self.quantization == "8bit"),
                )
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["device_map"] = self.device
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name, **load_kwargs
            )
        else:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self._processor = BlipProcessor.from_pretrained(self.model_name)
            self._model = BlipForConditionalGeneration.from_pretrained(
                self.model_name
            ).to(self.device)

        self._model.eval()

    # ── Inference ────────────────────────────────────────────────

    def generate(self, image: Image.Image, prompt: str | None = None) -> str:
        """Generate a caption for a single image. Returns ``""`` on failure.

        Args:
            image: PIL Image (any size — the processor handles resize).
            prompt: Optional text prompt. For BLIP-2 only; ignored by BLIP
                base (which always uses ``CAPTION_PROMPT``).

        Returns:
            Caption string, whitespace-stripped. Empty string when the
            model is missing, the input is unreadable, or generation
            fails — a warning is logged but no exception is raised so
            the metadata pipeline keeps moving.
        """
        if self._model is None:
            self._load()

        try:
            import torch

            is_blip2 = "blip2" in self.model_name.lower()
            if prompt is not None and is_blip2:
                inputs = self._processor(
                    images=image, text=prompt, return_tensors="pt"
                ).to(self._model.device)
            else:
                inputs = self._processor(
                    images=image, return_tensors="pt"
                ).to(self._model.device)

            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    min_length=self.min_length,
                )
            text = self._processor.decode(out[0], skip_special_tokens=True)
            return text.strip()
        except Exception as exc:  # noqa: BLE001 — graceful degradation
            logger.warning("Caption generation failed: %s", exc)
            return ""

    # ── Resource cleanup ─────────────────────────────────────────

    def close(self) -> None:
        """Release GPU memory by dropping the model + processor refs."""
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# ── Factory ────────────────────────────────────────────────────────


def build_caption_fn(
    model_name: str = "Salesforce/blip-image-captioning-base",
    device: str = "cpu",
    max_new_tokens: int = 50,
    min_length: int = 8,
    quantization: str = "none",
) -> "CaptionFn":
    """Build a :data:`CaptionFn` wired to a lazy-loaded :class:`CaptionGenerator`.

    Mirrors the lazy-load pattern from
    :func:`aic26.pipeline.metadata.build_metadata_callables`: the
    generator is ``None`` at construction, the model loads on the first
    image, and errors return ``""`` so the metadata pipeline never breaks.

    Returns:
        Closure ``(image) -> str``. Pass directly as
        ``MetadataExtractor(caption_fn=...)``.
    """
    gen = CaptionGenerator(
        model_name=model_name,
        device=device,
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        quantization=quantization,
    )

    def _caption_fn(image: Image.Image) -> str:
        # Default to the BLIP base prompt unless the caller overrides.
        return gen.generate(image, prompt=CAPTION_PROMPT)

    return _caption_fn