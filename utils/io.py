"""Centralized JSON I/O helpers for DimScan dataset and job files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_parent_dir(path: str | Path) -> Path:
    """Ensure a path's parent directory exists and return the normalized path."""
    normalized_path = Path(path)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    return normalized_path


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    """Write JSON data to a file and return the final path."""
    normalized_path = ensure_parent_dir(path)

    with normalized_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, sort_keys=False)
        file.write("\n")

    return normalized_path


def write_json_atomic(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    """Write JSON data through a temporary sibling file before replacing the target."""
    normalized_path = ensure_parent_dir(path)
    tmp_path = normalized_path.with_suffix(normalized_path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, sort_keys=False)
        file.write("\n")

    tmp_path.replace(normalized_path)
    return normalized_path


def read_json(path: str | Path) -> Any:
    """Read and parse JSON from a file."""
    normalized_path = Path(path)

    with normalized_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_json_if_exists(path: str | Path, default: Any = None) -> Any:
    """Read JSON from a file if it exists, otherwise return the provided default."""
    normalized_path = Path(path)
    if not normalized_path.exists():
        return default
    return read_json(normalized_path)


def file_exists(path: str | Path) -> bool:
    """Return True when the path exists and is a file."""
    return Path(path).is_file()


def dir_exists(path: str | Path) -> bool:
    """Return True when the path exists and is a directory."""
    return Path(path).is_dir()


def utc_now_iso() -> str:
    """Return the current UTC timestamp as a JSON-friendly ISO string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
