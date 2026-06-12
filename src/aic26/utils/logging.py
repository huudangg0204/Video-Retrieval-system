"""
Centralized logging configuration using loguru.

Provides structured logging with rotation, retention, and formatting.
Replaces scattered print() statements across the codebase.

Usage:
    from aic26.utils.logging import get_logger
    
    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.error("Failed to load model: {}", error_msg)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: str = "structured",
) -> None:
    """
    Configure logging for the entire application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to write logs to file
        log_format: 'structured' for detailed output, 'simple' for minimal output
    
    Example:
        setup_logging(level="DEBUG", log_file=Path("app.log"))
    """
    # Remove all existing handlers
    logger.remove()
    
    if log_format == "structured":
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{module}:{function}:{line} | "
            "{message}"
        )
    else:
        console_format = "<level>{level: <8}</level> | <level>{message}</level>"
        file_format = "{time} | {level: <8} | {message}"
    
    # Console handler (stderr for better compatibility)
    logger.add(
        sys.stderr,
        level=level,
        format=console_format,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # File handler (if configured)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_file),
            level="DEBUG",  # Always log DEBUG to file
            format=file_format,
            rotation="10 MB",  # Rotate when file reaches 10 MB
            retention="30 days",  # Keep logs for 30 days
            compression="zip",  # Compress rotated logs
            backtrace=True,
            diagnose=True,
            enqueue=True,  # Thread-safe file writing
        )
    
    logger.info(f"Logging configured | Level: {level} | File: {log_file or 'console only'}")


def get_logger(name: str = __name__):
    """
    Get a logger instance bound with the module name.
    
    Args:
        name: Usually __name__ of the calling module
    
    Returns:
        A loguru logger with module context
    
    Example:
        logger = get_logger(__name__)
        logger.info("Processing image {}", image_path)
    """
    return logger.bind(module=name)


# ── Convenience functions for common patterns ──────────────────────────

def log_function_call(logger_instance, func_name: str, **kwargs):
    """Log function entry with arguments (for debugging)."""
    args_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger_instance.debug(f"→ {func_name}({args_str})")


def log_duration(logger_instance, func_name: str, duration_seconds: float):
    """Log function execution duration."""
    if duration_seconds < 1.0:
        logger_instance.debug(f"← {func_name} completed in {duration_seconds*1000:.1f}ms")
    elif duration_seconds < 60:
        logger_instance.debug(f"← {func_name} completed in {duration_seconds:.2f}s")
    else:
        minutes = duration_seconds / 60
        logger_instance.info(f"← {func_name} completed in {minutes:.1f}min")
