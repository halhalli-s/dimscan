"""Build the runtime SKU catalogue JSON from vendor Excel sheets."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from metadata.sku_catalogue import parse_catalogue_spec_pot_prior, pot_prior_available


COLUMN_ALIASES = {
    "ITEM #": "raw_sku",
    "ITEM": "raw_sku",
    "ITEM#": "raw_sku",
    "SKU": "raw_sku",
    "CATEGORY": "category",
    "DESCRIPTION": "description",
    "COMMON NAME": "common_name",
    "COMMON_NAME": "common_name",
    "SPEC": "spec",
    "UOM": "uom",
    "UNIT PRICE": "unit_price",
    "UNIT_PRICE": "unit_price",
    "SOURCEFILE": "source_file",
    "SOURCE FILE": "source_file",
    "SOURCE SHEET": "source_sheet",
    "SOURCESHEET": "source_sheet",
}
RECOGNIZED_COLUMNS = {
    "raw_sku",
    "category",
    "description",
    "common_name",
    "spec",
    "uom",
    "unit_price",
    "source_file",
    "source_sheet",
}
XLSX_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {
    "main": XLSX_MAIN_NS,
    "rel": XLSX_REL_NS,
    "pkgrel": PACKAGE_REL_NS,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalized_column_name(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = " ".join(text.replace("_", " ").split())
    if not text:
        return ""
    return COLUMN_ALIASES.get(text, text.lower().replace(" ", "_"))


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def normalize_raw_sku(value: Any) -> tuple[str | None, str | None]:
    raw_text = clean_text(value)
    if raw_text is None:
        return None, None

    cleaned = raw_text
    try:
        decimal = Decimal(cleaned)
    except InvalidOperation:
        pass
    else:
        if decimal == decimal.to_integral_value():
            cleaned = str(decimal.quantize(Decimal("1")))
        else:
            cleaned = format(decimal.normalize(), "f")

    cleaned = cleaned.strip()
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]
    if not cleaned:
        return raw_text, None
    return raw_text, cleaned


def clean_unit_price(value: Any) -> float | str | None:
    text = clean_text(value)
    if text is None:
        return None
    try:
        return float(str(text).replace("$", "").replace(",", ""))
    except ValueError:
        return text


def completeness_score(entry: dict[str, Any]) -> int:
    fields = ("category", "description", "common_name", "spec", "uom", "unit_price")
    score = sum(1 for field in fields if entry.get(field) not in (None, ""))
    if pot_prior_available(entry.get("pot_prior")):
        score += 2
    return score


def normalized_for_conflict(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        key: entry.get(key)
        for key in ("category", "description", "common_name", "spec", "uom", "unit_price")
        if entry.get(key) not in (None, "")
    }


def entry_from_row(row: dict[str, Any], source_path: Path, sheet_name: str | None = None) -> tuple[dict[str, Any] | None, bool]:
    raw_sku, sku = normalize_raw_sku(row.get("raw_sku"))
    if sku is None:
        return None, True

    source_file = clean_text(row.get("source_file")) or source_path.name
    spec = clean_text(row.get("spec"))
    pot_prior = parse_catalogue_spec_pot_prior(spec)
    return (
        {
            "sku": sku,
            "raw_sku": raw_sku,
            "category": clean_text(row.get("category")),
            "description": clean_text(row.get("description")),
            "common_name": clean_text(row.get("common_name")),
            "spec": spec,
            "uom": clean_text(row.get("uom")),
            "unit_price": clean_unit_price(row.get("unit_price")),
            "source_file": source_file,
            "pot_prior": pot_prior,
        },
        False,
    )


def xml_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(child.text or "" for child in element.iter() if child.tag == f"{{{XLSX_MAIN_NS}}}t")


def load_shared_strings(zip_file: ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zip_file.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [xml_text(item) for item in root.findall("main:si", NS)]


def column_index(cell_ref: str) -> int:
    letters = "".join(character for character in cell_ref if character.isalpha())
    index = 0
    for character in letters:
        index = index * 26 + ord(character.upper()) - 64
    return max(0, index - 1)


def cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return xml_text(cell.find("main:is", NS))

    value = cell.find("main:v", NS)
    if value is None:
        return ""
    raw_value = value.text or ""
    if cell_type == "s" and raw_value:
        return shared_strings[int(raw_value)]
    return raw_value


def read_sheet_rows(zip_file: ZipFile, sheet_path: str, shared_strings: list[str]) -> list[list[str]]:
    root = ET.fromstring(zip_file.read(sheet_path))
    rows: list[list[str]] = []
    for row_element in root.findall(".//main:sheetData/main:row", NS):
        values: list[str] = []
        for cell in row_element.findall("main:c", NS):
            index = column_index(cell.attrib.get("r", "A1"))
            while len(values) <= index:
                values.append("")
            values[index] = cell_value(cell, shared_strings)
        rows.append(values)
    return rows


def workbook_sheets(path: Path) -> list[dict[str, str]]:
    with ZipFile(path) as zip_file:
        workbook = ET.fromstring(zip_file.read("xl/workbook.xml"))
        rels = ET.fromstring(zip_file.read("xl/_rels/workbook.xml.rels"))
        relmap = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("pkgrel:Relationship", NS)
        }
        sheets = []
        for sheet in workbook.findall("main:sheets/main:sheet", NS):
            rel_id = sheet.attrib[f"{{{XLSX_REL_NS}}}id"]
            target = relmap[rel_id]
            sheet_path = target.lstrip("/")
            if not sheet_path.startswith("xl/"):
                sheet_path = f"xl/{sheet_path}"
            sheets.append({"name": sheet.attrib["name"], "path": sheet_path})
        return sheets


def nonempty_row(row: list[Any]) -> bool:
    return any(clean_text(value) is not None for value in row)


def detect_header_row(rows: list[list[Any]]) -> tuple[int | None, list[str], dict[str, str]]:
    for row_index, row in enumerate(rows[:25]):
        columns = [normalized_column_name(value) for value in row]
        detected = {
            str(original).strip(): normalized
            for original, normalized in zip(row, columns)
            if normalized in RECOGNIZED_COLUMNS
        }
        recognized = {column for column in columns if column in RECOGNIZED_COLUMNS}
        if "raw_sku" in recognized and len(recognized - {"source_file", "source_sheet"}) >= 3:
            return row_index, columns, detected
    return None, [], {}


def row_to_dict(row: list[Any], columns: list[str]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for index, column in enumerate(columns):
        if not column:
            continue
        value = row[index] if index < len(row) else None
        if column in RECOGNIZED_COLUMNS:
            record[column] = value
    return record


def read_excel_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read all data-like sheets from an .xlsx workbook using stdlib XML parsing."""
    all_records: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "workbook_path": str(path),
        "sheet_names": [],
        "sheets_read": [],
        "sheets_skipped": [],
        "rows_read_by_sheet": {},
        "valid_sku_rows_by_sheet": {},
        "detected_columns_by_sheet": {},
        "skip_reason_by_sheet": {},
    }

    with ZipFile(path) as zip_file:
        shared_strings = load_shared_strings(zip_file)
        sheets = workbook_sheets(path)
        diagnostics["sheet_names"] = [sheet["name"] for sheet in sheets]
        for sheet in sheets:
            sheet_name = sheet["name"]
            rows = read_sheet_rows(zip_file, sheet["path"], shared_strings)
            header_index, columns, detected_columns = detect_header_row(rows)
            diagnostics["detected_columns_by_sheet"][sheet_name] = detected_columns

            if header_index is None:
                diagnostics["sheets_skipped"].append(sheet_name)
                diagnostics["rows_read_by_sheet"][sheet_name] = 0
                diagnostics["valid_sku_rows_by_sheet"][sheet_name] = 0
                diagnostics["skip_reason_by_sheet"][sheet_name] = "missing_item_number_column"
                continue

            data_rows = [row for row in rows[header_index + 1 :] if nonempty_row(row)]
            records = [row_to_dict(row, columns) for row in data_rows]
            valid_sku_rows = sum(1 for record in records if normalize_raw_sku(record.get("raw_sku"))[1] is not None)

            if valid_sku_rows == 0:
                diagnostics["sheets_skipped"].append(sheet_name)
                diagnostics["rows_read_by_sheet"][sheet_name] = len(data_rows)
                diagnostics["valid_sku_rows_by_sheet"][sheet_name] = 0
                diagnostics["skip_reason_by_sheet"][sheet_name] = "no_valid_sku_rows"
                continue

            diagnostics["sheets_read"].append(sheet_name)
            diagnostics["rows_read_by_sheet"][sheet_name] = len(data_rows)
            diagnostics["valid_sku_rows_by_sheet"][sheet_name] = valid_sku_rows
            for record in records:
                record["_source_sheet"] = sheet_name
            all_records.extend(records)

    return all_records, diagnostics


def merge_entries(inputs: list[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    duplicate_skus: set[str] = set()
    conflicting_duplicate_skus: set[str] = set()
    total_rows_read = 0
    rows_with_missing_sku = 0
    workbook_diagnostics: list[dict[str, Any]] = []

    for input_path in inputs:
        rows, diagnostics = read_excel_rows(input_path)
        workbook_diagnostics.append(diagnostics)
        total_rows_read += len(rows)
        for row in rows:
            entry, missing_sku = entry_from_row(row, input_path, clean_text(row.get("_source_sheet")))
            if missing_sku:
                rows_with_missing_sku += 1
                continue
            assert entry is not None
            sku = entry["sku"]
            existing = items.get(sku)
            if existing is None:
                items[sku] = entry
                continue

            duplicate_skus.add(sku)
            if normalized_for_conflict(existing) != normalized_for_conflict(entry):
                conflicting_duplicate_skus.add(sku)
            if completeness_score(entry) > completeness_score(existing):
                items[sku] = entry

    report = build_report(
        inputs=inputs,
        items=items,
        total_rows_read=total_rows_read,
        rows_with_missing_sku=rows_with_missing_sku,
        duplicate_skus=duplicate_skus,
        conflicting_duplicate_skus=conflicting_duplicate_skus,
        workbook_diagnostics=workbook_diagnostics,
    )
    return items, report


def build_report(
    *,
    inputs: list[Path],
    items: dict[str, dict[str, Any]],
    total_rows_read: int,
    rows_with_missing_sku: int,
    duplicate_skus: set[str],
    conflicting_duplicate_skus: set[str],
    workbook_diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    pot_prior_available_count = 0
    parsed_round_pot_count = 0
    parsed_rectangular_count = 0
    gallon_unmapped_count = 0
    unparsed_specs: list[str] = []

    for item in items.values():
        prior = item.get("pot_prior") if isinstance(item.get("pot_prior"), dict) else {}
        if pot_prior_available(prior):
            pot_prior_available_count += 1
        if prior.get("shape") == "round" and prior.get("available"):
            parsed_round_pot_count += 1
        if prior.get("shape") == "rectangular" and prior.get("available"):
            parsed_rectangular_count += 1
        notes = prior.get("notes") if isinstance(prior.get("notes"), list) else []
        if "gallon_spec_not_mapped" in notes:
            gallon_unmapped_count += 1
        if "spec_unparsed" in notes and prior.get("raw_spec"):
            unparsed_specs.append(str(prior["raw_spec"]))

    return {
        "input_files": [str(path) for path in inputs],
        "workbooks": workbook_diagnostics,
        "total_rows_read": total_rows_read,
        "rows_with_missing_sku": rows_with_missing_sku,
        "final_item_count": len(items),
        "duplicate_sku_count": len(duplicate_skus),
        "duplicate_skus": sorted(duplicate_skus),
        "conflicting_duplicate_skus": sorted(conflicting_duplicate_skus),
        "pot_prior_available_count": pot_prior_available_count,
        "pot_prior_unavailable_count": len(items) - pot_prior_available_count,
        "parsed_round_pot_count": parsed_round_pot_count,
        "parsed_rectangular_count": parsed_rectangular_count,
        "gallon_unmapped_count": gallon_unmapped_count,
        "unparsed_spec_count": len(unparsed_specs),
        "sample_unparsed_specs": sorted(set(unparsed_specs))[:25],
        "generated_at": utc_now_iso(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def build_catalogue(inputs: list[Path], output: Path, report_path: Path) -> dict[str, Any]:
    items, report = merge_entries(inputs)
    generated_at = report["generated_at"]
    catalogue = {
        "catalogue_schema_version": "v1.0",
        "source": "nursery_sku_catalog",
        "generated_at": generated_at,
        "item_count": len(items),
        "items": dict(sorted(items.items())),
    }
    write_json(output, catalogue)
    write_json(report_path, report)
    return {"catalogue": catalogue, "report": report}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Vendor Excel source file. Repeatable.")
    parser.add_argument("--output", required=True, help="Runtime SKU catalogue JSON to create or replace.")
    parser.add_argument("--report", required=True, help="Merge report JSON to create or replace.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [Path(value) for value in args.input]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {', '.join(missing)}")
    result = build_catalogue(inputs, Path(args.output), Path(args.report))
    report = result["report"]
    print(
        "Built SKU catalogue: "
        f"{report['final_item_count']} items, "
        f"{report['duplicate_sku_count']} duplicate SKUs, "
        f"{report['pot_prior_available_count']} pot priors available."
    )


if __name__ == "__main__":
    main()
