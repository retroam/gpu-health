"""Deterministic XID extraction and catalog lookup helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

XID_PATTERNS = [
    re.compile(r"\bXid\s*\([^)]*\)\s*:\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bXID\s*[:=]?\s*(\d+)\b", re.IGNORECASE),
]

PCI_PATTERNS = [
    re.compile(r"PCI:([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}(?:\.[0-9a-f])?)", re.IGNORECASE),
    re.compile(r"(\b[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]\b)", re.IGNORECASE),
]

GPU_MODEL_PATTERNS = [
    re.compile(r"\b(GB200|B200|B100|H100|A100|V100|L40S|L40|RTX\s*\d{3,4})\b", re.IGNORECASE),
]


@dataclass
class XidEntry:
    xid_code: int
    mnemonic: str
    description: str
    immediate_action: str
    investigatory_action: str
    severity: str
    trigger_conditions: str
    applies_to: dict[str, bool]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "XidEntry":
        return cls(
            xid_code=int(data.get("xid_code", -1)),
            mnemonic=str(data.get("mnemonic", "Unknown")),
            description=str(data.get("description", "Unknown XID")),
            immediate_action=str(data.get("immediate_action", "UNKNOWN")),
            investigatory_action=str(data.get("investigatory_action", "UNKNOWN")),
            severity=str(data.get("severity", "Unknown")),
            trigger_conditions=str(data.get("trigger_conditions", "")),
            applies_to=dict(data.get("applies_to", {})),
        )


def extract_xid_code(log_text: str) -> int | None:
    """Extract an NVIDIA XID code from noisy log text."""
    for pattern in XID_PATTERNS:
        match = pattern.search(log_text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def extract_pci_devices(log_text: str) -> list[str]:
    """Extract PCI identifiers that appear in logs."""
    found: list[str] = []
    for pattern in PCI_PATTERNS:
        for match in pattern.finditer(log_text):
            pci = match.group(1)
            if pci not in found:
                found.append(pci)
    return found


def extract_gpu_models(log_text: str) -> list[str]:
    """Extract GPU model mentions that appear in logs."""
    found: list[str] = []
    for pattern in GPU_MODEL_PATTERNS:
        for match in pattern.finditer(log_text):
            model = re.sub(r"\s+", "", match.group(1).upper())
            if model not in found:
                found.append(model)
    return found


def load_catalog(catalog_path: str | Path) -> dict[int, XidEntry]:
    """Load catalog JSON and return map keyed by xid_code."""
    path = Path(catalog_path)
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    catalog: dict[int, XidEntry] = {}
    for row in rows:
        entry = XidEntry.from_dict(row)
        if entry.xid_code >= 0:
            catalog[entry.xid_code] = entry
    return catalog


def build_heuristic_summary(log_text: str, entry: XidEntry | None, xid_code: int | None) -> str:
    """Generate a grounded plain-text summary without an LLM."""
    if xid_code is None:
        return (
            "No NVIDIA XID code was detected in the provided log. "
            "Check whether this is an XID-related incident or provide additional lines around NVRM messages."
        )

    if entry is None:
        return (
            f"Detected XID {xid_code}, but it was not found in the local catalog. "
            "Update data/xid_catalog.json from NVIDIA's latest XID catalog before triage."
        )

    pci_devices = extract_pci_devices(log_text)
    gpu_models = extract_gpu_models(log_text)
    pci_hint = f" Affected PCI IDs: {', '.join(pci_devices)}." if pci_devices else ""
    gpu_hint = f" GPU models mentioned: {', '.join(gpu_models)}." if gpu_models else ""
    trigger_hint = f" Trigger context: {entry.trigger_conditions}" if entry.trigger_conditions else ""

    return (
        f"This log shows XID {entry.xid_code} ({entry.description}). "
        f"NVIDIA severity: {entry.severity}. "
        f"Immediate action: {entry.immediate_action}. "
        f"Investigatory action: {entry.investigatory_action}."
        f"{pci_hint}{gpu_hint}{trigger_hint}"
    )
