"""General parsing helpers for DimScan metadata and arrangement inputs."""

from __future__ import annotations

import re
from typing import Any


ARRANGEMENT_RE = re.compile(r"^[1-9][0-9]*x[1-9][0-9]*$")
DIRECT_POT_RE = re.compile(
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>inches|inch|in|\"|cm|mm)\b",
    re.IGNORECASE,
)
VOLUME_POT_RE = re.compile(
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>quarts|quart|qt|gallons|gallon|gal|liters|liter|litres|litre|l)\b",
    re.IGNORECASE,
)


def parse_positive_int(value: Any, *, name: str) -> int:
    """Parse a positive integer from metadata input."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc

    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def parse_nonnegative_int(value: Any, *, name: str) -> int:
    """Parse a non-negative integer from metadata input."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc

    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return parsed


def parse_positive_float(value: Any, *, name: str) -> float:
    """Parse a positive float from metadata input."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number.") from exc

    if parsed <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return parsed


def normalize_arrangement_type(value: str) -> str:
    """Normalize and validate an arrangement string such as '2 x 5'."""
    normalized = re.sub(r"\s+", "", value.strip().lower())
    if ARRANGEMENT_RE.fullmatch(normalized) is None:
        raise ValueError(f"Invalid arrangement type: {value!r}")
    return normalized


def parse_arrangement_dims(arrangement_type: str) -> tuple[int, int]:
    """Return arrangement dimensions as rows and columns."""
    normalized = normalize_arrangement_type(arrangement_type)
    rows_text, cols_text = normalized.split("x", maxsplit=1)
    return int(rows_text), int(cols_text)


def arrangement_cell_count(arrangement_type: str) -> int:
    """Return the number of cells in an arrangement."""
    rows, cols = parse_arrangement_dims(arrangement_type)
    return rows * cols


def is_square_arrangement(arrangement_type: str) -> bool:
    """Return True when an arrangement has equal rows and columns."""
    rows, cols = parse_arrangement_dims(arrangement_type)
    return rows == cols


def is_single_arrangement(arrangement_type: str) -> bool:
    """Return True only for a 1x1 arrangement."""
    return normalize_arrangement_type(arrangement_type) == "1x1"


def _empty_pot_prior(*, source: str, raw_value: Any = None, notes: list[str] | None = None) -> dict[str, Any]:
    return {
        "pot_prior_available": False,
        "source": source,
        "diameter_in": None,
        "height_in": None,
        "volume_qt": None,
        "confidence": "none",
        "raw_value": raw_value,
        "raw_unit": None,
        "notes": notes or [],
    }


def _inches(value: float, unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"inch", "inches", "in", '"'}:
        return value
    if normalized == "cm":
        return value / 2.54
    if normalized == "mm":
        return value / 25.4
    raise ValueError(f"Unsupported length unit: {unit!r}")


def _volume_qt(value: float, unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"qt", "quart", "quarts"}:
        return value
    if normalized in {"gal", "gallon", "gallons"}:
        return value * 4.0
    if normalized in {"l", "liter", "liters", "litre", "litres"}:
        return value * 1.05668821
    raise ValueError(f"Unsupported volume unit: {unit!r}")


def parse_pot_prior(
    metadata: dict[str, Any] | None,
    *,
    container_lookup: dict[str, dict[str, float]] | None = None,
    source: str = "sku_exact",
) -> dict[str, Any]:
    """Normalize optional SKU pot evidence without making it required."""
    if not isinstance(metadata, dict):
        return _empty_pot_prior(source="missing")

    lookup = container_lookup or {}
    notes: list[str] = []
    raw_candidates: list[tuple[str, Any]] = []
    for key in (
        "standard_pot_diameter_in",
        "pot_diameter_in",
        "container_diameter_in",
        "diameter_in",
    ):
        if metadata.get(key) not in (None, ""):
            raw_candidates.append((key, metadata.get(key)))
    for key in ("container_size", "pot_size", "nursery_pot_size", "size", "description"):
        if metadata.get(key) not in (None, ""):
            raw_candidates.append((key, metadata.get(key)))

    if not raw_candidates:
        return _empty_pot_prior(source="missing", notes=["no_pot_prior_text_in_sku_metadata"])

    for key, raw_value in raw_candidates:
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool) and float(raw_value) > 0:
            return {
                "pot_prior_available": True,
                "source": source,
                "diameter_in": float(raw_value),
                "height_in": None,
                "volume_qt": None,
                "confidence": "high",
                "raw_value": raw_value,
                "raw_unit": "in",
                "notes": [f"parsed_direct_numeric:{key}"],
            }

        text = str(raw_value).strip()
        direct = DIRECT_POT_RE.search(text)
        if direct:
            diameter = _inches(float(direct.group("value")), direct.group("unit"))
            return {
                "pot_prior_available": True,
                "source": source,
                "diameter_in": diameter,
                "height_in": None,
                "volume_qt": None,
                "confidence": "high",
                "raw_value": raw_value,
                "raw_unit": direct.group("unit"),
                "notes": [f"parsed_direct_length:{key}"],
            }

        volume = VOLUME_POT_RE.search(text)
        if volume:
            volume_qt = _volume_qt(float(volume.group("value")), volume.group("unit"))
            best_key = None
            best_delta = None
            for lookup_key, lookup_value in lookup.items():
                lookup_volume = lookup_value.get("volume_qt")
                if lookup_volume is None:
                    continue
                delta = abs(float(lookup_volume) - volume_qt)
                if best_delta is None or delta < best_delta:
                    best_key = lookup_key
                    best_delta = delta
            if best_key is not None:
                matched = lookup[best_key]
                return {
                    "pot_prior_available": True,
                    "source": "sku_container_lookup",
                    "diameter_in": matched.get("diameter_in"),
                    "height_in": matched.get("height_in"),
                    "volume_qt": volume_qt,
                    "confidence": "medium" if best_delta is not None and best_delta <= 1.0 else "low",
                    "raw_value": raw_value,
                    "raw_unit": volume.group("unit"),
                    "notes": [f"matched_container_lookup:{best_key}", "volume_not_directly_converted_to_diameter"],
                }
            notes.append(f"volume_found_without_lookup_match:{key}")
        else:
            notes.append(f"unparseable_pot_prior:{key}={text}")

    return _empty_pot_prior(source="unparseable", raw_value=raw_candidates[0][1], notes=notes)
