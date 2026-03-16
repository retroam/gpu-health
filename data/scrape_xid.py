"""Scrape NVIDIA's XID catalog into normalized JSON."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_URL = "https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html"
DEFAULT_TARGET_GPUS = ("A100", "H100", "B100", "B200", "GB200")

GPU_HEADER_RE = re.compile(r"\b([abghv]\d{2,4}|gb\d{3,4}|rtx\s*\d{3,4}|l\d{2,3})\b", re.IGNORECASE)


@dataclass
class XidRecord:
    xid_code: int
    mnemonic: str
    description: str
    applies_to: dict[str, bool]
    immediate_action: str
    investigatory_action: str
    severity: str
    trigger_conditions: str


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_header(value: str) -> str:
    value = normalize_text(value).lower()
    value = value.replace("\u2013", "-")
    return value


def extract_int(value: str) -> int | None:
    match = re.search(r"\b(\d{1,4})\b", value)
    if not match:
        return None
    return int(match.group(1))


def parse_bool(value: str) -> bool | None:
    token = normalize_text(value).lower()
    if token in {"yes", "true", "y", "1", "x"}:
        return True
    if token in {"no", "false", "n", "0", ""}:
        return False
    return None


def pick_table(soup):
    candidates = []
    for table in soup.find_all("table"):
        headers = [normalize_header(cell.get_text(" ", strip=True)) for cell in table.find_all("th")]
        if not headers:
            continue
        if any("xid" in header for header in headers):
            row_count = len(table.find_all("tr"))
            candidates.append((row_count, table))

    if not candidates:
        return None

    return sorted(candidates, key=lambda pair: pair[0], reverse=True)[0][1]


def parse_headers(header_cells: Iterable) -> list[str]:
    headers: list[str] = []
    for cell in header_cells:
        headers.append(normalize_header(cell.get_text(" ", strip=True)))
    return headers


def extract_xid_code(row: dict[str, str]) -> int | None:
    for key in ("code", "xid code", "xid"):
        value = row.get(key)
        if value is None:
            continue
        xid_code = extract_int(value)
        if xid_code is not None:
            return xid_code

    for header, value in row.items():
        normalized = normalize_header(header)
        if normalized in {"type (xid)", "xid 154 linkage"}:
            continue
        if normalized == "code" or normalized.endswith(" code"):
            xid_code = extract_int(value)
            if xid_code is not None:
                return xid_code

    return None


def parse_record(headers: list[str], values: list[str]) -> XidRecord | None:
    row = {headers[idx]: values[idx] for idx in range(min(len(headers), len(values)))}

    xid_code = extract_xid_code(row)
    if xid_code is None:
        return None

    mnemonic = next((v for k, v in row.items() if "mnemonic" in k), "")
    description = next((v for k, v in row.items() if "description" in k), "")
    immediate_action = next((v for k, v in row.items() if "immediate" in k and "action" in k), "")
    investigatory_action = next((v for k, v in row.items() if "investigatory" in k and "action" in k), "")
    severity = next((v for k, v in row.items() if "severity" in k), "")
    trigger_conditions = next(
        (v for k, v in row.items() if "trigger" in k or "condition" in k or "notes" in k), ""
    )

    applies_to: dict[str, bool] = {}
    for header, value in row.items():
        if header in {
            "xid",
            "xid code",
            "xid error",
            "mnemonic",
            "description",
            "severity",
            "immediate action",
            "investigatory action",
            "trigger conditions",
            "notes",
        }:
            continue

        matches_gpu = bool(GPU_HEADER_RE.search(header)) or "applies" in header
        if not matches_gpu:
            continue

        parsed = parse_bool(value)
        if parsed is not None:
            applies_to[header.upper()] = parsed
        elif normalize_text(value):
            applies_to[header.upper()] = True

    return XidRecord(
        xid_code=xid_code,
        mnemonic=normalize_text(mnemonic),
        description=normalize_text(description),
        applies_to=applies_to,
        immediate_action=normalize_text(immediate_action),
        investigatory_action=normalize_text(investigatory_action),
        severity=normalize_text(severity),
        trigger_conditions=normalize_text(trigger_conditions),
    )


def scrape_xid_catalog(url: str) -> list[XidRecord]:
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("requests and beautifulsoup4 are required; install gpu-health requirements first") from exc

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    if response.apparent_encoding:
        response.encoding = response.apparent_encoding

    soup = BeautifulSoup(response.text, "html.parser")
    table = pick_table(soup)
    if table is None:
        raise RuntimeError("Could not find XID table on page")

    rows = table.find_all("tr")
    if not rows:
        return []

    header_cells = rows[0].find_all(["th", "td"])
    headers = parse_headers(header_cells)

    records: list[XidRecord] = []
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        values = [normalize_text(cell.get_text(" ", strip=True)) for cell in cells]
        if not any(values):
            continue
        record = parse_record(headers, values)
        if record is None:
            continue
        records.append(record)

    records.sort(key=lambda item: item.xid_code)
    return records


def is_applicable(record: XidRecord, target_gpus: list[str]) -> bool:
    if not record.applies_to:
        return True

    target_set = {gpu.upper() for gpu in target_gpus}
    matching_values = [enabled for gpu, enabled in record.applies_to.items() if gpu.upper() in target_set]
    if matching_values:
        return any(matching_values)

    return any(record.applies_to.values())


def main() -> None:
    try:
        parser = argparse.ArgumentParser(description="Scrape NVIDIA XID catalog")
        parser.add_argument("--url", default=DEFAULT_URL)
        parser.add_argument("--out", default="xid_catalog.json", help="Output JSON path")
        parser.add_argument(
            "--include-unused",
            action="store_true",
            help="Keep rows whose mnemonic/description contains 'unused'",
        )
        parser.add_argument(
            "--target-gpu",
            action="append",
            default=list(DEFAULT_TARGET_GPUS),
            help="Target GPU family to keep when applicability columns are present",
        )
        args = parser.parse_args()

        records = scrape_xid_catalog(args.url)

        if not args.include_unused:
            records = [
                r
                for r in records
                if "unused" not in r.mnemonic.lower() and "unused" not in r.description.lower()
            ]

        records = [record for record in records if is_applicable(record, args.target_gpu)]

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump([asdict(record) for record in records], handle, ensure_ascii=True, indent=2)

        print(f"Wrote {len(records)} rows to {out_path}")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
