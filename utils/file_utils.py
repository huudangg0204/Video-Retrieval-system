"""utils/file_utils.py — Filesystem helpers."""
import os
from typing import Optional


def ensure_dir(path: str) -> None:
    """Create *path* (and parents) if it doesn't already exist."""
    if path:
        os.makedirs(path, exist_ok=True)


def safe_basename(path: str) -> str:
    """Return os.path.basename(path), falling back to str(path) on error."""
    try:
        return os.path.basename(path)
    except Exception:
        return str(path)
