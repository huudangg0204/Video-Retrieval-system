"""
Unit tests for configuration settings.
"""
from __future__ import annotations

import pytest
from pathlib import Path
import paddle
print("GPU available:", paddle.device.is_compiled_with_cuda())
print("cuDNN version:", paddle.device.get_cudnn_version())