"""Simple deterministic box suggestion rules for DimScan development."""

from __future__ import annotations

from math import ceil
from typing import Any


def round_up_to_increment(value: float, *, increment: float = 1.0) -> float:
    """Round a positive value up to the nearest positive increment."""
    if value <= 0:
        raise ValueError("value must be positive.")
    if increment <= 0:
        raise ValueError("increment must be positive.")
    return float(ceil(value / increment) * increment)


def add_padding(value: float, *, padding_in: float = 1.0) -> float:
    """Add non-negative padding to a positive dimension."""
    if value <= 0:
        raise ValueError("value must be positive.")
    if padding_in < 0:
        raise ValueError("padding_in must be non-negative.")
    return value + padding_in


def suggest_box_from_candidates(
    *,
    length_candidate_in: float | None,
    width_candidate_in: float | None,
    height_candidate_in: float | None,
    padding_in: float = 1.0,
    round_increment_in: float = 1.0,
) -> dict[str, Any]:
    """Suggest a padded, rounded box from candidate dimensions."""
    if length_candidate_in is None or width_candidate_in is None or height_candidate_in is None:
        raise ValueError("All candidate dimensions are required.")

    length = round_up_to_increment(
        add_padding(length_candidate_in, padding_in=padding_in),
        increment=round_increment_in,
    )
    width = round_up_to_increment(
        add_padding(width_candidate_in, padding_in=padding_in),
        increment=round_increment_in,
    )
    height = round_up_to_increment(
        add_padding(height_candidate_in, padding_in=padding_in),
        increment=round_increment_in,
    )

    return {
        "length_in": length,
        "width_in": width,
        "height_in": height,
        "method": "rules_padding_rounding",
        "padding_in": padding_in,
        "round_increment_in": round_increment_in,
    }


def suggest_box_from_combined_features(
    combined_features: dict[str, Any],
    *,
    padding_in: float = 1.0,
    round_increment_in: float = 1.0,
) -> dict[str, Any]:
    """Suggest a box from a combined features dictionary."""
    features = combined_features.get("ai2_features", {})
    if not isinstance(features, dict) or not features:
        features = combined_features.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("combined_features must contain a feature dictionary.")

    return suggest_box_from_candidates(
        length_candidate_in=features.get("object_length_in"),
        width_candidate_in=features.get("object_width_in"),
        height_candidate_in=features.get("object_height_in"),
        padding_in=padding_in,
        round_increment_in=round_increment_in,
    )
