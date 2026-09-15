"""Small logging helpers for DimScan modules."""

from __future__ import annotations

import logging
from pathlib import Path


FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "dimscan") -> logging.Logger:
    """Return a configured standard-library logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(FORMAT))
        logger.addHandler(handler)

    return logger


def configure_file_logging(
    log_path: str | Path,
    *,
    logger_name: str = "dimscan",
) -> logging.Logger:
    """Add file logging to a DimScan logger."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = get_logger(logger_name)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(handler)
    return logger
