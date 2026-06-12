"""
Filesystem helper utilities.

Provides safe file operations with proper error handling.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Create directory and all parent directories if they don't exist.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object of the created directory
        
    Example:
        ensure_dir("./data/embeddings/video1")
    """
    path = Path(path) if not isinstance(path, Path) else path
    if path and not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


def safe_basename(path: Union[str, Path]) -> str:
    """
    Get the basename of a path safely (never raises).
    
    Args:
        path: File or directory path
        
    Returns:
        Basename string, or str(path) on error
        
    Example:
        safe_basename("/path/to/file.txt")  # → "file.txt"
        safe_basename(None)                  # → "None"
    """
    try:
        return os.path.basename(str(path))
    except Exception:
        return str(path)


def get_file_stem(path: Union[str, Path]) -> str:
    """
    Get filename without extension safely.
    
    Args:
        path: File path
        
    Returns:
        Filename stem
        
    Example:
        get_file_stem("/path/to/frame_042.jpg")  # → "frame_042"
    """
    try:
        return Path(path).stem
    except Exception:
        return str(path)


def find_files(
    directory: Union[str, Path],
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    sort: bool = True,
) -> list[Path]:
    """
    Find all files with given extensions in a directory.
    
    Args:
        directory: Directory to search
        extensions: File extensions to match
        sort: Whether to sort results alphabetically
        
    Returns:
        List of Path objects
        
    Example:
        images = find_files("./keyframes", extensions=(".jpg", ".png"))
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"*{ext.upper()}"))
    
    if sort:
        files.sort()
    
    return files