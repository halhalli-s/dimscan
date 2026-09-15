"""Local SKU metadata lookup helpers for DimScan item records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from capture.sku import require_valid_sku
from app.config import DimScanConfig
from metadata.parser import parse_pot_prior
from metadata.sku_catalogue import (
    default_catalogue_path,
    lookup_catalogue_sku,
    parse_catalogue_spec_pot_prior,
)
from utils.io import read_json


def normalize_metadata_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow metadata copy without mutating caller data."""
    metadata = dict(record)
    if isinstance(metadata.get("pot_prior"), dict):
        pass
    elif metadata.get("spec") not in (None, ""):
        metadata["pot_prior"] = parse_catalogue_spec_pot_prior(metadata.get("spec"))
    else:
        metadata["pot_prior"] = parse_pot_prior(
            metadata,
            container_lookup=DimScanConfig.nursery_container_lookup,
        )
    return metadata


def _record_sku(record: dict[str, Any]) -> str | None:
    for key in ("sku", "SKU", "item_sku", "Item SKU"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return require_valid_sku(value)
    return None


def _normalize_mapping_keys(mapping: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in mapping.items():
        cleaned_sku = require_valid_sku(key)
        normalized[cleaned_sku] = value
    return normalized


def lookup_sku_from_mapping(sku: str, mapping: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Look up SKU metadata from an in-memory mapping."""
    cleaned_sku = require_valid_sku(sku)
    record = mapping.get(cleaned_sku)
    if record is None:
        record = mapping.get(cleaned_sku.upper())

    if record is None:
        return {
            "sku": cleaned_sku,
            "known": False,
        }

    metadata = normalize_metadata_record(record)
    metadata["sku"] = cleaned_sku
    metadata["known"] = True
    return metadata


def load_sku_mapping(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a SKU metadata mapping from a local JSON file."""
    data = read_json(path)
    if isinstance(data, list):
        mapping = {
            sku: item
            for item in data
            if isinstance(item, dict)
            for sku in [_record_sku(item)]
            if sku is not None
        }
        if len(mapping) != len(data):
            raise ValueError("SKU catalog list entries must be objects with a sku field.")
        return mapping

    if not isinstance(data, dict):
        raise ValueError("SKU mapping JSON must be an object or list.")

    mapping = data.get("items") if "items" in data else data
    if isinstance(mapping, list):
        return load_sku_mapping_from_records(mapping)
    if not isinstance(mapping, dict):
        raise ValueError("SKU mapping must be an object or contain items as an object/list.")

    for key, value in mapping.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise ValueError("SKU mapping keys must be strings and values must be objects.")

    return _normalize_mapping_keys(mapping)


def load_sku_mapping_from_records(records: list[Any]) -> dict[str, dict[str, Any]]:
    """Load a SKU mapping from a list of record dictionaries."""
    mapping: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("SKU catalog list entries must be objects.")
        sku = _record_sku(record)
        if sku is None:
            raise ValueError("SKU catalog list entries must include a sku field.")
        mapping[sku] = record
    return mapping


def lookup_sku_from_file(sku: str, path: str | Path) -> dict[str, Any]:
    """Look up SKU metadata from a local JSON mapping file."""
    mapping = load_sku_mapping(path)
    return lookup_sku_from_mapping(sku, mapping)


def default_sku_catalog_path() -> Path:
    """Return the bundled local SKU catalog path."""
    return default_catalogue_path()


def lookup_sku(sku: str, catalog_path: str | Path | None = None) -> dict[str, Any]:
    """Look up SKU metadata from the default or provided local catalog."""
    cleaned_sku = require_valid_sku(sku)
    catalogue_lookup = lookup_catalogue_sku(cleaned_sku, catalog_path)
    if catalogue_lookup["status"] == "found" and isinstance(catalogue_lookup.get("item"), dict):
        metadata = normalize_metadata_record(catalogue_lookup["item"])
        metadata.update(
            {
                "sku": cleaned_sku,
                "known": True,
                "lookup_status": "found",
                "catalogue": catalogue_lookup.get("catalogue"),
            }
        )
        return metadata

    path = Path(catalog_path) if catalog_path is not None else default_sku_catalog_path()
    if not path.exists():
        return {
            "sku": cleaned_sku,
            "known": False,
            "lookup_status": "not_found",
            "pot_prior": catalogue_lookup.get("pot_prior"),
            "catalogue": catalogue_lookup.get("catalogue"),
        }
    metadata = lookup_sku_from_file(cleaned_sku, path)
    if not metadata.get("known"):
        metadata["lookup_status"] = "not_found"
        metadata["pot_prior"] = catalogue_lookup.get("pot_prior")
        metadata["catalogue"] = catalogue_lookup.get("catalogue")
    else:
        metadata["lookup_status"] = "found"
    return metadata


def make_item(
    sku: str,
    quantity: int = 1,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a normalized item record for DimScan job item lists."""
    cleaned_sku = require_valid_sku(sku)
    if not isinstance(quantity, int) or quantity <= 0:
        raise ValueError(f"Invalid item quantity: {quantity!r}")

    return {
        "sku": cleaned_sku,
        "quantity": quantity,
        "metadata": metadata or {"sku": cleaned_sku, "known": False},
    }


def make_item_from_sku(
    sku: str,
    quantity: int = 1,
    catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create an item record by normalizing a SKU and looking up local metadata."""
    cleaned_sku = require_valid_sku(sku)
    metadata = lookup_sku(cleaned_sku, catalog_path=catalog_path)
    return make_item(cleaned_sku, quantity, metadata)
