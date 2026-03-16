"""Evaluate predicted responses on held-out XID datasets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEMO_DIR = ROOT / "demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from lib.tinker_runtime import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    TinkerSampler,
    default_runtime_manifest_path,
    load_runtime_config,
)
from xid_lookup import XidEntry, build_heuristic_summary

XID_PATTERNS = [
    re.compile(r"\bXid\s*\([^)]*\)\s*:\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bXID\s*[:=]?\s*(\d+)\b", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model outputs for XID triage")
    parser.add_argument("--dataset", required=True, help="JSONL dataset with meta.xid_code")
    parser.add_argument(
        "--predictions",
        help="JSONL predictions aligned by row index; if omitted, predictions are generated in-process",
    )
    parser.add_argument(
        "--predictor",
        choices=("tinker", "heuristic"),
        default="tinker",
        help="Predictor used when --predictions is omitted",
    )
    parser.add_argument("--output-predictions", help="Optional path to write generated predictions JSONL")
    parser.add_argument(
        "--tinker-manifest",
        default=str(default_runtime_manifest_path(ROOT, "gpu-health-v1")),
        help="Path to runtime manifest created by train/train.py",
    )
    parser.add_argument("--tinker-model-path", help="Explicit Tinker checkpoint path")
    parser.add_argument("--tinker-base-model", help="Fallback base model for Tinker sampling")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--report", default="eval_report.json")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_xid(text: str) -> int | None:
    for pattern in XID_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def extract_prediction_text(row: dict) -> str:
    if isinstance(row.get("output"), str):
        return row["output"]
    if isinstance(row.get("assistant_response"), str):
        return row["assistant_response"]
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                return message["content"]
    return ""


def extract_reference_text(row: dict) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                return message["content"]
    return ""


def build_generation_messages(row: dict) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return []
    if messages and messages[-1].get("role") == "assistant":
        return messages[:-1]
    return messages


def extract_log_text(row: dict) -> str:
    for message in reversed(build_generation_messages(row)):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def action_bucket_match(text: str, action: str) -> bool:
    if not action:
        return False
    compact_text = re.sub(r"[^A-Z0-9]+", "", text.upper())
    compact_action = re.sub(r"[^A-Z0-9]+", "", action.upper())
    return bool(compact_action) and compact_action in compact_text


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[idx : idx + order]) for idx in range(len(tokens) - order + 1))


def sentence_bleu(reference: str, candidate: str, *, max_order: int = 2) -> float:
    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(candidate)
    if not candidate_tokens:
        return 0.0

    precisions: list[float] = []
    for order in range(1, max_order + 1):
        cand_ngrams = ngrams(candidate_tokens, order)
        if not cand_ngrams:
            precisions.append(0.0)
            continue

        ref_ngrams = ngrams(reference_tokens, order)
        overlap = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
        precisions.append((overlap + 1.0) / (sum(cand_ngrams.values()) + 1.0))

    if len(candidate_tokens) > len(reference_tokens):
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - (len(reference_tokens) / max(len(candidate_tokens), 1)))

    return brevity_penalty * math.exp(sum(math.log(max(precision, 1e-9)) for precision in precisions) / max_order)


def build_meta_entry(meta: dict) -> XidEntry:
    return XidEntry(
        xid_code=int(meta.get("xid_code", -1)),
        mnemonic=str(meta.get("mnemonic", "Unknown")),
        description=str(meta.get("description", "Unknown XID")),
        immediate_action=str(meta.get("immediate_action", "UNKNOWN")),
        investigatory_action=str(meta.get("investigatory_action", "UNKNOWN")),
        severity=str(meta.get("severity", "Unknown")),
        trigger_conditions=str(meta.get("trigger_conditions", "")),
        applies_to={},
    )


def generate_predictions(dataset: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.predictor == "heuristic":
        predictions: list[dict] = []
        for row in dataset:
            meta = row.get("meta", {})
            log_text = extract_log_text(row)
            entry = build_meta_entry(meta if isinstance(meta, dict) else {})
            summary = build_heuristic_summary(log_text=log_text, entry=entry, xid_code=entry.xid_code)
            predictions.append({"output": summary})
        return predictions

    config = load_runtime_config(
        manifest_path=args.tinker_manifest if Path(args.tinker_manifest).exists() else None,
        model_path=args.tinker_model_path,
        base_model=args.tinker_base_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    sampler = TinkerSampler(config)
    predictions = []
    for row in dataset:
        predictions.append({"output": sampler.sample_messages(build_generation_messages(row))})
    return predictions


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def run(args: argparse.Namespace) -> None:
    dataset = load_jsonl(Path(args.dataset))
    predictions = load_jsonl(Path(args.predictions)) if args.predictions else generate_predictions(dataset, args)

    if not args.predictions and args.output_predictions:
        write_jsonl(Path(args.output_predictions), predictions)

    if len(dataset) != len(predictions):
        raise RuntimeError("Dataset and predictions must have equal number of rows")

    total = len(dataset)
    xid_exact = 0
    immediate_exact = 0
    investigatory_exact = 0
    all_action_exact = 0
    bleu_scores: list[float] = []

    for idx, expected_row in enumerate(dataset):
        pred_row = predictions[idx]
        prediction_text = extract_prediction_text(pred_row)
        predicted_xid = extract_xid(prediction_text)

        meta = expected_row.get("meta", {})
        expected_xid = int(meta.get("xid_code", -1))
        if predicted_xid == expected_xid:
            xid_exact += 1

        expected_immediate = str(meta.get("immediate_action", ""))
        expected_investigatory = str(meta.get("investigatory_action", ""))
        immediate_match = action_bucket_match(prediction_text, expected_immediate)
        investigatory_match = action_bucket_match(prediction_text, expected_investigatory)
        if immediate_match:
            immediate_exact += 1
        if investigatory_match:
            investigatory_exact += 1
        if immediate_match and investigatory_match:
            all_action_exact += 1

        reference_text = extract_reference_text(expected_row)
        if reference_text:
            bleu_scores.append(sentence_bleu(reference_text, prediction_text))

    report = {
        "rows": total,
        "xid_exact_matches": xid_exact,
        "xid_exact_accuracy": (xid_exact / total) if total else 0.0,
        "immediate_action_matches": immediate_exact,
        "immediate_action_accuracy": (immediate_exact / total) if total else 0.0,
        "investigatory_action_matches": investigatory_exact,
        "investigatory_action_accuracy": (investigatory_exact / total) if total else 0.0,
        "all_action_matches": all_action_exact,
        "all_action_accuracy": (all_action_exact / total) if total else 0.0,
        "summary_bleu": (sum(bleu_scores) / len(bleu_scores)) if bleu_scores else 0.0,
        "summary_bleu_rows": len(bleu_scores),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=True, indent=2))
    print(f"report={report_path}")


def main() -> None:
    try:
        args = parse_args()
        run(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
