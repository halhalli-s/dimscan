"""Arrangement helpers for DimScan view planning and quantity validation."""

from __future__ import annotations

from typing import Any

from metadata.parser import (
    arrangement_cell_count,
    is_single_arrangement,
    is_square_arrangement,
    normalize_arrangement_type,
    parse_arrangement_dims,
)


def infer_view_mode(arrangement_type: str) -> str:
    """Infer a view mode from an arrangement type."""
    normalized = normalize_arrangement_type(arrangement_type)
    if is_single_arrangement(normalized) or is_square_arrangement(normalized):
        return "single_view"
    return "two_view_rectangle"


def default_view_names(arrangement_type: str, view_mode: str | None = None) -> list[str]:
    """Return default view names for an arrangement and view mode."""
    normalized = normalize_arrangement_type(arrangement_type)
    selected_view_mode = view_mode or infer_view_mode(normalized)

    if selected_view_mode == "single_view":
        return ["view_01"]
    if selected_view_mode == "two_view_rectangle":
        return ["view_01", "view_02"]
    raise ValueError(f"Unsupported view mode: {selected_view_mode!r}")


def arrangement_summary(arrangement_type: str) -> dict[str, Any]:
    """Return normalized arrangement details for scan planning."""
    normalized = normalize_arrangement_type(arrangement_type)
    rows, cols = parse_arrangement_dims(normalized)
    view_mode = infer_view_mode(normalized)
    return {
        "arrangement_type": normalized,
        "rows": rows,
        "cols": cols,
        "cell_count": rows * cols,
        "is_single": is_single_arrangement(normalized),
        "is_square": is_square_arrangement(normalized),
        "view_mode": view_mode,
        "default_view_names": default_view_names(normalized, view_mode),
    }


def validate_item_quantities_for_arrangement(
    items: list[dict[str, Any]],
    arrangement_type: str,
    *,
    strict: bool = True,
) -> int:
    """Validate item quantities against arrangement cell count."""
    total_quantity = 0
    for item in items:
        quantity = item.get("quantity")
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            raise ValueError(f"Invalid item quantity: {quantity!r}")
        total_quantity += quantity

    expected_quantity = arrangement_cell_count(arrangement_type)
    if strict and total_quantity != expected_quantity:
        raise ValueError(
            f"Total item quantity {total_quantity} does not match arrangement "
            f"cell count {expected_quantity} for {normalize_arrangement_type(arrangement_type)!r}."
        )

    return total_quantity


def make_arrangement_record(
    arrangement_type: str,
    *,
    items: list[dict[str, Any]] | None = None,
    strict_quantity: bool = True,
) -> dict[str, Any]:
    """Create a normalized arrangement record for a job."""
    record = arrangement_summary(arrangement_type)

    if items is None:
        record["quantity_validated"] = False
        record["total_quantity"] = None
        return record

    record["quantity_validated"] = True
    record["total_quantity"] = validate_item_quantities_for_arrangement(
        items,
        record["arrangement_type"],
        strict=strict_quantity,
    )
    return record
