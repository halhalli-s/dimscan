"""SKU normalization and validation helpers for manual or scanner input."""

from __future__ import annotations

import re


SKU_ALLOWED_RE = re.compile(r"^[A-Z0-9_.-]+$")


def normalize_sku(raw_sku: str) -> str:
    """Normalize a raw SKU string into the DimScan canonical SKU form."""
    cleaned = raw_sku.strip()
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = cleaned.upper()
    cleaned = re.sub(r"[^A-Z0-9_.-]", "", cleaned)
    return cleaned


def is_valid_sku(sku: str, *, min_length: int = 1, max_length: int = 128) -> bool:
    """Return whether a SKU uses allowed characters and length."""
    if not sku:
        return False
    if len(sku) < min_length or len(sku) > max_length:
        return False
    return SKU_ALLOWED_RE.fullmatch(sku) is not None


def require_valid_sku(raw_sku: str) -> str:
    """Normalize a SKU and raise ValueError if it is invalid."""
    cleaned = normalize_sku(raw_sku)
    if not is_valid_sku(cleaned):
        raise ValueError(f"Invalid SKU: {raw_sku!r}")
    return cleaned


def parse_scanned_sku(raw_input: str) -> str:
    """Parse scanner input into a validated SKU."""
    return require_valid_sku(raw_input)
