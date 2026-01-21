"""
Logging utilities for the data construction pipeline.

Provides a logger that outputs to both console and file simultaneously.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "pipeline",
    log_dir: Optional[Path] = None,
    log_filename: str = "pipeline.log",
    level: int = logging.INFO,
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
) -> logging.Logger:
    """
    Setup a logger that outputs to both console and file.

    Args:
        name: Logger name
        log_dir: Directory for log file. If None, only console logging is enabled.
        log_filename: Name of the log file
        level: Base logging level
        console_level: Console handler level (defaults to `level`)
        file_level: File handler level (defaults to `level`)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Formatter for both handlers
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler - always enabled
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level or level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler - only if log_dir is provided
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_filename

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(file_level or level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Log the log file path
        logger.info(f"Log file: {log_path}")

    return logger


def get_logger(name: str = "pipeline") -> logging.Logger:
    """
    Get an existing logger by name.

    Args:
        name: Logger name

    Returns:
        Logger instance (may not be configured if setup_logger wasn't called)
    """
    return logging.getLogger(name)


def reset_logger(name: str = "pipeline") -> None:
    """
    Reset a logger by removing all its handlers.

    Args:
        name: Logger name to reset
    """
    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
