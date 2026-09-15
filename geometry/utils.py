"""Basic numeric geometry helpers for DimScan dictionary-based geometry."""

from __future__ import annotations

from typing import Iterable


def positive_float(value: float, *, name: str) -> float:
    """Convert a value to float and require it to be positive."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive.")
    numeric_value = float(value)
    if numeric_value <= 0:
        raise ValueError(f"{name} must be positive.")
    return numeric_value


def optional_positive_float(value: float | None, *, name: str) -> float | None:
    """Validate an optional positive float."""
    if value is None:
        return None
    return positive_float(value, name=name)


def max_or_none(values: Iterable[float | None]) -> float | None:
    """Return the maximum non-None value, or None when no values are present."""
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return max(valid_values)


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Return numerator divided by denominator when both are usable."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator
