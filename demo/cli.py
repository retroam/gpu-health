"""CLI for hybrid GPU XID triage (deterministic lookup + summary)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.tinker_runtime import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from triage import (
    default_catalog_path,
    default_tinker_manifest_path,
    diagnose_log,
)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:  # pragma: no cover
    Console = None
    Panel = None
    Table = None


def _read_log_input(log_arg: str | None, log_file: str | None) -> str:
    if log_arg:
        return log_arg
    if log_file:
        return Path(log_file).read_text(encoding="utf-8")
    return input("Paste log snippet: ").strip()


def _render_plain(payload: dict[str, object]) -> None:
    print(f"XID: {payload['xid_code']}")
    print(f"Description: {payload['description']}")
    print(f"Severity: {payload['severity']}")
    print(f"Immediate Action: {payload['immediate_action']}")
    print(f"Investigatory Action: {payload['investigatory_action']}")
    if payload.get("pci_devices"):
        print(f"PCI Devices: {', '.join(payload['pci_devices'])}")
    if payload.get("gpu_models"):
        print(f"GPU Models: {', '.join(payload['gpu_models'])}")
    print(f"Summary Mode: {payload['summary_mode']}")
    print()
    print(payload["summary"])
    if payload.get("summary_warning"):
        print()
        print(f"Warning: {payload['summary_warning']}")


def _render_rich(payload: dict[str, object]) -> None:
    if Console is None or Table is None or Panel is None:
        _render_plain(payload)
        return

    console = Console()
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("XID", str(payload["xid_code"]))
    table.add_row("Description", str(payload["description"]))
    table.add_row("Severity", str(payload["severity"]))
    table.add_row("Immediate", str(payload["immediate_action"]))
    table.add_row("Investigatory", str(payload["investigatory_action"]))
    if payload.get("pci_devices"):
        table.add_row("PCI Devices", ", ".join(payload["pci_devices"]))
    if payload.get("gpu_models"):
        table.add_row("GPU Models", ", ".join(payload["gpu_models"]))
    table.add_row("Summary Mode", str(payload["summary_mode"]))
    console.print(Panel(table, title="GPU Health Triage", expand=False))
    console.print(Panel(str(payload["summary"]), title="Summary", expand=False))
    if payload.get("summary_warning"):
        console.print(Panel(str(payload["summary_warning"]), title="Warning", expand=False))


def run() -> None:
    parser = argparse.ArgumentParser(description="Diagnose NVIDIA XID logs")
    parser.add_argument("log", nargs="?", help="Raw log snippet")
    parser.add_argument("--log-file", help="Path to log file")
    parser.add_argument("--catalog", default=str(default_catalog_path(ROOT)), help="Path to xid_catalog.json")
    parser.add_argument(
        "--summary-mode",
        choices=("auto", "heuristic", "openai", "tinker"),
        default="auto",
        help="How to produce text summary",
    )
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model for --summary-mode=openai")
    parser.add_argument(
        "--tinker-manifest",
        default=str(default_tinker_manifest_path(ROOT)),
        help="Path to runtime manifest created by train/train.py",
    )
    parser.add_argument("--tinker-model-path", help="Explicit Tinker checkpoint path")
    parser.add_argument("--tinker-base-model", help="Fallback base model if no checkpoint manifest is available")
    parser.add_argument("--tinker-max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--tinker-temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--json", action="store_true", help="Emit JSON payload")
    args = parser.parse_args()

    log_text = _read_log_input(args.log, args.log_file)
    payload = diagnose_log(
        log_text=log_text,
        root=ROOT,
        catalog_path=args.catalog,
        summary_mode=args.summary_mode,
        openai_model=args.openai_model,
        tinker_manifest=args.tinker_manifest,
        tinker_model_path=args.tinker_model_path,
        tinker_base_model=args.tinker_base_model,
        tinker_max_tokens=args.tinker_max_tokens,
        tinker_temperature=args.tinker_temperature,
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return

    _render_rich(payload)


if __name__ == "__main__":
    try:
        run()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
