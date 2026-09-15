"""Simple validators for DimScan geometry and feature dictionaries."""

from __future__ import annotations

from typing import Any


def require_keys(data: dict[str, Any], keys: list[str], *, context: str = "data") -> None:
    """Raise ValueError when required keys are missing."""
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Missing required keys in {context}: {', '.join(missing)}")


def get_nested(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    """Safely read a nested dictionary value."""
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def as_float_or_none(value: Any) -> float | None:
    """Convert numeric values to float, returning None for missing or non-numeric values."""
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def as_bool_available(value: Any) -> bool:
    """Return True when a value is available."""
    return value is not None
