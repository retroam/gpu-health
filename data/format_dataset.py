"""Convert synthetic logs to SFT JSONL and create eval splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a GPU cluster health diagnostic assistant. Given a log snippet, identify "
    "the NVIDIA XID error, state its severity and NVIDIA's recommended immediate and "
    "investigatory actions, and summarize any additional context from surrounding lines."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format generated examples into train/eval splits")
    parser.add_argument("--input", default="synthetic_logs.jsonl")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--train-xid-max", type=int, default=70)
    parser.add_argument(
        "--journalctl-train-frac",
        type=float,
        default=0.0,
        help="Fraction of journalctl samples to keep in train split",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_messages(row: dict) -> list[dict[str, str]]:
    if isinstance(row.get("messages"), list) and row["messages"]:
        return row["messages"]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(row.get("log", ""))},
        {"role": "assistant", "content": str(row.get("assistant_response", ""))},
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def run(args: argparse.Namespace) -> None:
    source = load_jsonl(Path(args.input))

    train_rows: list[dict] = []
    eval_xid_rows: list[dict] = []
    eval_format_rows: list[dict] = []

    journalctl_train_budget = int(sum(1 for row in source if str(row.get("log_source", "")).lower() == "journalctl") * args.journalctl_train_frac)
    journalctl_train_used = 0

    for row in source:
        xid_code = int(row.get("xid_code", -1))
        log_source = str(row.get("log_source", "unknown")).lower()

        formatted = {
            "messages": build_messages(row),
            "meta": {
                "xid_code": xid_code,
                "mnemonic": row.get("mnemonic"),
                "description": row.get("description"),
                "log_source": log_source,
                "gpu_model": row.get("gpu_model"),
                "severity": row.get("severity"),
                "immediate_action": row.get("immediate_action"),
                "investigatory_action": row.get("investigatory_action"),
                "trigger_conditions": row.get("trigger_conditions"),
            },
        }

        if xid_code > args.train_xid_max:
            eval_xid_rows.append(formatted)
            continue

        if log_source == "journalctl":
            if journalctl_train_used < journalctl_train_budget:
                journalctl_train_used += 1
                train_rows.append(formatted)
            else:
                eval_format_rows.append(formatted)
            continue

        train_rows.append(formatted)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    eval_xid_path = out_dir / "eval_xid.jsonl"
    eval_format_path = out_dir / "eval_format.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(eval_xid_path, eval_xid_rows)
    write_jsonl(eval_format_path, eval_format_rows)

    print(f"train: {len(train_rows)} -> {train_path}")
    print(f"eval_xid: {len(eval_xid_rows)} -> {eval_xid_path}")
    print(f"eval_format: {len(eval_format_rows)} -> {eval_format_path}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
