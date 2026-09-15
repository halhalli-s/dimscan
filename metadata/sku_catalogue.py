"""SKU catalogue loading and vendor SPEC-text parsing helpers."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from capture.sku import require_valid_sku
from utils.io import read_json


ROUND_SPEC_RE = re.compile(
    r"(?P<diameter>[0-9]+(?:\.[0-9]+)?)\s*(?:\"|in|inch|inches)\s*"
    r"(?P<label>pot|basket|hb|hanging basket)?\b",
    re.IGNORECASE,
)
RECT_SPEC_RE = re.compile(
    r"(?P<width>[0-9]+(?:\.[0-9]+)?)\s*x\s*(?P<depth>[0-9]+(?:\.[0-9]+)?)\s*"
    r"(?P<label>flat|tray)?\b",
    re.IGNORECASE,
)
GALLON_SPEC_RE = re.compile(r"\b[0-9]+(?:\.[0-9]+)?\s*(?:gal|gallon|gallons)\b", re.IGNORECASE)


def empty_catalogue_pot_prior(
    *,
    raw_spec: Any = None,
    source: str = "sku_catalogue",
    confidence: str = "unknown",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Return an unavailable catalogue-style pot prior."""
    return {
        "available": False,
        "shape": "unknown",
        "diameter_in": None,
        "width_in": None,
        "depth_in": None,
        "height_in": None,
        "raw_spec": raw_spec,
        "source": source,
        "confidence": confidence,
        "notes": notes or [],
    }


def parse_catalogue_spec_pot_prior(spec: Any) -> dict[str, Any]:
    """Parse vendor-style SPEC text into a normalized pot prior."""
    if spec is None or str(spec).strip() == "":
        return empty_catalogue_pot_prior(raw_spec=None, notes=["spec_missing"])

    raw_spec = str(spec).strip()
    normalized = re.sub(r"\s+", " ", raw_spec)

    rectangular = RECT_SPEC_RE.search(normalized)
    if rectangular:
        width = float(rectangular.group("width"))
        depth = float(rectangular.group("depth"))
        return {
            "available": True,
            "shape": "rectangular",
            "diameter_in": None,
            "width_in": width,
            "depth_in": depth,
            "height_in": None,
            "raw_spec": raw_spec,
            "source": "sku_catalogue",
            "confidence": "catalog",
            "notes": ["rectangular_dimensions_parsed_from_spec"],
        }

    round_spec = ROUND_SPEC_RE.search(normalized)
    if round_spec:
        diameter = float(round_spec.group("diameter"))
        return {
            "available": True,
            "shape": "round",
            "diameter_in": diameter,
            "width_in": diameter,
            "depth_in": diameter,
            "height_in": None,
            "raw_spec": raw_spec,
            "source": "sku_catalogue",
            "confidence": "catalog",
            "notes": ["diameter_parsed_from_spec", "width_depth_derived_from_diameter"],
        }

    if GALLON_SPEC_RE.search(normalized):
        return empty_catalogue_pot_prior(raw_spec=raw_spec, notes=["gallon_spec_not_mapped"])

    return empty_catalogue_pot_prior(raw_spec=raw_spec, notes=["spec_unparsed"])


def pot_prior_available(pot_prior: dict[str, Any] | None) -> bool:
    """Return availability for either catalogue-style or legacy pot priors."""
    if not isinstance(pot_prior, dict):
        return False
    return bool(pot_prior.get("available") or pot_prior.get("pot_prior_available"))


def pot_prior_model_confidence(pot_prior: dict[str, Any] | None) -> str:
    """Map catalogue/legacy confidence into the existing model-prior gate."""
    if not isinstance(pot_prior, dict):
        return "none"
    confidence = str(pot_prior.get("confidence") or "none")
    if confidence == "catalog":
        return "high"
    return confidence


def default_catalogue_prior_for_missing_sku() -> dict[str, Any]:
    """Return the endpoint prior for an unknown SKU."""
    return empty_catalogue_pot_prior(notes=["sku_not_found"])


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_catalogue_path() -> Path:
    """Return the preferred runtime catalogue path, with old-name fallback."""
    preferred = Path(__file__).with_name("sku_catalogue.json")
    if preferred.exists():
        return preferred
    return Path(__file__).with_name("sku_catalog.json")


@lru_cache(maxsize=4)
def load_catalogue(path_text: str | None = None) -> dict[str, Any]:
    """Load a catalogue JSON once and cache it by path."""
    path = Path(path_text) if path_text else default_catalogue_path()
    if not path.exists():
        return {
            "catalogue_schema_version": "missing",
            "source": "missing",
            "item_count": 0,
            "items": {},
            "status": "missing",
            "path": str(path),
        }
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"SKU catalogue must contain a JSON object: {path}")
    items = data.get("items") if isinstance(data.get("items"), dict) else data
    if not isinstance(items, dict):
        raise ValueError(f"SKU catalogue items must be an object: {path}")
    return {
        **data,
        "items": items,
        "item_count": int(data.get("item_count") or len(items)),
        "status": "loaded",
        "path": str(path),
    }


def lookup_catalogue_sku(sku: str, path: str | Path | None = None) -> dict[str, Any]:
    """Return a status-oriented lookup response for the runtime catalogue."""
    cleaned_sku = require_valid_sku(sku)
    catalogue = load_catalogue(str(path) if path is not None else None)
    items = catalogue.get("items") if isinstance(catalogue.get("items"), dict) else {}
    item = items.get(cleaned_sku) or items.get(cleaned_sku.upper())
    if item is None:
        cleaned_lower = cleaned_sku.lower()
        for item_sku, candidate in items.items():
            if isinstance(item_sku, str) and item_sku.lower() == cleaned_lower:
                item = candidate
                break
    if isinstance(item, dict):
        pot_prior = item.get("pot_prior") if isinstance(item.get("pot_prior"), dict) else None
        if pot_prior is None and item.get("spec") not in (None, ""):
            pot_prior = parse_catalogue_spec_pot_prior(item.get("spec"))
            item = {**item, "pot_prior": pot_prior}
        if pot_prior is None:
            pot_prior = empty_catalogue_pot_prior(raw_spec=None, notes=["spec_missing"])
        return {
            "status": "found",
            "sku": cleaned_sku,
            "item": item,
            "pot_prior": pot_prior,
            "catalogue": {
                "schema_version": catalogue.get("catalogue_schema_version"),
                "source": catalogue.get("source"),
                "path": catalogue.get("path"),
                "item_count": catalogue.get("item_count"),
            },
        }
    return {
        "status": "not_found",
        "sku": cleaned_sku,
        "item": None,
        "pot_prior": default_catalogue_prior_for_missing_sku(),
        "catalogue": {
            "schema_version": catalogue.get("catalogue_schema_version"),
            "source": catalogue.get("source"),
            "path": catalogue.get("path"),
            "item_count": catalogue.get("item_count"),
        },
        "warning": "SKU not found in catalogue",
    }
