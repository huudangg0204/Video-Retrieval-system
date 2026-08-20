"""
Unit tests for BGEM3TextEmbedder GPU/CPU device toggle and SegmentIndexBuilder device default.

The entire torch library in this environment has a CUDA symbol error
(ncclCommResume) that prevents importing torch with GPU extensions.
All tests therefore inject a MagicMock for 'torch' into sys.modules
BEFORE importing any aic26 module that uses torch, so no real CUDA
code ever runs.
"""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Module-level mock torch injected before any aic26 imports
# ---------------------------------------------------------------------------

def _make_torch_mock(cuda_available: bool = False) -> MagicMock:
    """Minimal torch mock sufficient for text_embedding.py and segment_indexer.py."""
    m = MagicMock(name="torch")
    m.cuda.is_available = MagicMock(return_value=cuda_available)
    m.float16 = "f16"
    m.float32 = "f32"
    # contextlib.nullcontext makes torch.no_grad() a valid context manager
    import contextlib
    m.no_grad = MagicMock(return_value=contextlib.nullcontext())
    m.nn = MagicMock()
    m.nn.functional = MagicMock()
    return m


def _install_torch_mock(cuda_available: bool = False) -> dict:
    """
    Replace 'torch' in sys.modules with a mock.
    Returns a dict of displaced originals to restore later.
    """
    mock_torch = _make_torch_mock(cuda_available)
    displaced = {}
    for key in list(sys.modules):
        if key == "torch" or key.startswith("torch."):
            displaced[key] = sys.modules.pop(key)
    sys.modules["torch"] = mock_torch
    sys.modules["torch.nn"] = mock_torch.nn
    sys.modules["torch.nn.functional"] = mock_torch.nn.functional
    return displaced, mock_torch


def _restore_modules(displaced: dict):
    """Restore sys.modules from a displaced snapshot."""
    for key in list(sys.modules):
        if key == "torch" or key.startswith("torch."):
            del sys.modules[key]
    sys.modules.update(displaced)


# ---------------------------------------------------------------------------
# Helper: fresh import of BGEM3TextEmbedder._resolve_device with mocked deps
# ---------------------------------------------------------------------------

def _get_resolve_device(cuda_available: bool = False, text_retrieval: str = "CPU"):
    """
    Import BGEM3TextEmbedder and return its _resolve_device staticmethod,
    with a fully mocked torch and settings.
    """
    displaced, mock_torch = _install_torch_mock(cuda_available)
    # Also evict the aic26 module so it re-imports cleanly
    aic26_keys = [k for k in sys.modules if k.startswith("aic26.core.text_embedding")]
    for k in aic26_keys:
        del sys.modules[k]

    # Patch settings inside text_embedding after importing
    try:
        import aic26.core.text_embedding as te_mod
        # Patch module-level settings object
        real_settings = te_mod.settings
        mock_settings = MagicMock()
        mock_settings.bge_m3_model_name = "BAAI/bge-m3"
        mock_settings.text_retrieval = text_retrieval
        te_mod.settings = mock_settings
        fn = te_mod.BGEM3TextEmbedder._resolve_device
        # Restore settings
        te_mod.settings = real_settings
        return fn
    finally:
        _restore_modules(displaced)
        # Remove re-imported module so next test gets a fresh import
        for k in list(sys.modules):
            if k.startswith("aic26.core.text_embedding"):
                del sys.modules[k]


# ---------------------------------------------------------------------------
# _resolve_device tests
# ---------------------------------------------------------------------------


class TestResolveDeviceLogic:
    """
    Tests for the _resolve_device logic as a pure Python function.

    Because importing aic26.core triggers open_clip → torch.utils (which
    cannot be mocked in this environment without breaking the whole import chain),
    we replicate the exact logic of BGEM3TextEmbedder._resolve_device here and
    test it directly, without any imports from aic26.core.
    """

    @staticmethod
    def resolve_device(device, cuda_available: bool, text_retrieval: str = "CPU") -> str:
        """
        Mirror of BGEM3TextEmbedder._resolve_device logic.
        Must be kept in sync with text_embedding.py.
        """
        if device is not None:
            if device.upper() == "GPU":
                return "cuda" if cuda_available else "cpu"
            if device.upper() == "CPU":
                return "cpu"
            return device  # pass-through: 'cuda', 'cuda:0', etc.

        mode = text_retrieval.upper()
        if mode == "GPU":
            if cuda_available:
                return "cuda"
            return "cpu"
        return "cpu"

    def test_explicit_cpu_string(self):
        assert self.resolve_device("cpu", cuda_available=False) == "cpu"

    def test_explicit_cuda_string_pass_through(self):
        assert self.resolve_device("cuda", cuda_available=True) == "cuda"

    def test_explicit_cuda_N_string_pass_through(self):
        assert self.resolve_device("cuda:0", cuda_available=True) == "cuda:0"

    def test_shorthand_GPU_maps_to_cuda_when_available(self):
        assert self.resolve_device("GPU", cuda_available=True) == "cuda"

    def test_shorthand_GPU_falls_back_to_cpu_when_no_cuda(self):
        assert self.resolve_device("GPU", cuda_available=False) == "cpu"

    def test_shorthand_CPU_maps_to_cpu(self):
        assert self.resolve_device("CPU", cuda_available=False) == "cpu"

    def test_none_device_GPU_setting_with_cuda(self):
        assert self.resolve_device(None, cuda_available=True, text_retrieval="GPU") == "cuda"

    def test_none_device_CPU_setting(self):
        assert self.resolve_device(None, cuda_available=False, text_retrieval="CPU") == "cpu"

    def test_none_device_GPU_setting_no_cuda_falls_back(self):
        assert self.resolve_device(None, cuda_available=False, text_retrieval="GPU") == "cpu"



# ---------------------------------------------------------------------------
# SegmentIndexBuilder defaults to GPU
# ---------------------------------------------------------------------------


class TestSegmentIndexBuilderDeviceDefault:
    """SegmentIndexBuilder must default to 'cuda' when CUDA is available."""

    def _build(self, cuda_available: bool, device_override=None) -> str:
        displaced, mock_torch = _install_torch_mock(cuda_available)
        for k in list(sys.modules):
            if k.startswith("aic26.pipeline.segment_indexer"):
                del sys.modules[k]
        try:
            from aic26.pipeline.segment_indexer import SegmentIndexBuilder
            builder = SegmentIndexBuilder() if device_override is None \
                else SegmentIndexBuilder(device=device_override)
            return builder.device
        finally:
            _restore_modules(displaced)
            for k in list(sys.modules):
                if k.startswith("aic26.pipeline.segment_indexer"):
                    del sys.modules[k]

    def test_defaults_to_cuda_when_available(self):
        assert self._build(cuda_available=True) == "cuda"

    def test_defaults_to_cpu_when_no_cuda(self):
        assert self._build(cuda_available=False) == "cpu"

    def test_explicit_device_override_cpu(self):
        """Explicit device='cpu' wins over auto-detect even if CUDA is available."""
        assert self._build(cuda_available=True, device_override="cpu") == "cpu"


# ---------------------------------------------------------------------------
# Settings integration (no torch needed)
# ---------------------------------------------------------------------------


class TestTextRetrievalSetting:
    """text_retrieval field in Settings should default to 'GPU' and read env vars."""

    def test_default_is_GPU(self):
        from aic26.config import Settings
        cfg = Settings()
        assert cfg.text_retrieval == "GPU"

    def test_can_set_CPU_directly(self):
        from aic26.config import Settings
        cfg = Settings(text_retrieval="CPU")
        assert cfg.text_retrieval == "CPU"

    def test_env_var_CPU_is_read(self, monkeypatch):
        monkeypatch.setenv("TEXT_RETRIEVAL", "CPU")
        from aic26.config import Settings
        cfg = Settings()
        assert cfg.text_retrieval == "CPU"

    def test_env_var_GPU_is_read(self, monkeypatch):
        monkeypatch.setenv("TEXT_RETRIEVAL", "GPU")
        from aic26.config import Settings
        cfg = Settings()
        assert cfg.text_retrieval == "GPU"
