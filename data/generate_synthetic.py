"""Generate synthetic XID log examples with OpenAI and write JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import random
import time
from pathlib import Path

SYSTEM_PROMPT = (
    "You generate realistic GPU incident logs and grounded operator responses for supervised fine-tuning. "
    "Never invent catalog actions beyond fields provided in prompt context. "
    "Return strict JSON only."
)

ASSISTANT_SYSTEM_PROMPT = (
    "You are a GPU cluster health diagnostic assistant. "
    "Given a log snippet, identify the NVIDIA XID error, state its severity and NVIDIA's "
    "recommended immediate and investigatory actions, and summarize useful context."
)

DEFAULT_GPU_CHOICES = ("A100", "H100", "B200", "GB200")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic log/response examples")
    parser.add_argument("--catalog", default="xid_catalog.json", help="Path to xid_catalog.json")
    parser.add_argument("--out", default="synthetic_logs.jsonl", help="Output JSONL path")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--examples-per-xid", type=int, default=6)
    parser.add_argument("--max-xids", type=int, default=0, help="0 means all rows")
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--request-timeout", type=float, default=60.0, help="Per-request timeout in seconds")
    parser.add_argument("--max-attempts", type=int, default=3, help="Attempts per XID before failing")
    parser.add_argument("--resume", action="store_true", help="Append missing examples to an existing output file")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def applicable_gpu_models(entry: dict[str, object]) -> list[str]:
    applies_to = entry.get("applies_to", {})
    if not isinstance(applies_to, dict):
        return list(DEFAULT_GPU_CHOICES)

    models = [str(model).strip().upper() for model, enabled in applies_to.items() if enabled]
    return models or list(DEFAULT_GPU_CHOICES)


def normalize_gpu_model(value: object, allowed_models: list[str]) -> str:
    candidate = str(value or "").strip().upper()
    if candidate in allowed_models:
        return candidate
    return random.choice(allowed_models)


def build_prompt(entry: dict[str, object], examples_per_xid: int, gpu_models: list[str]) -> str:
    gpu_model_list = ", ".join(gpu_models)
    return (
        "Generate synthetic Linux logs for one NVIDIA XID code.\n"
        "Return JSON with this schema:\n"
        "{{\"examples\": [{{\"gpu_model\": \"...\", \"log_source\": \"dmesg|journalctl\", \"log\": \"...\", \"assistant_response\": \"...\"}}]}}\n"
        "Constraints:\n"
        "- produce exactly {n} examples\n"
        "- include realistic timestamps, PCI bus IDs, and surrounding noisy lines\n"
        "- vary between dmesg and journalctl formats; if n >= 2, include both at least once\n"
        "- vary across these applicable GPU models when possible: {gpu_models}\n"
        "- assistant_response must be grounded only in provided fields\n\n"
        "Catalog entry:\n"
        f"xid_code: {entry.get('xid_code')}\n"
        f"mnemonic: {entry.get('mnemonic')}\n"
        f"description: {entry.get('description')}\n"
        f"severity: {entry.get('severity')}\n"
        f"immediate_action: {entry.get('immediate_action')}\n"
        f"investigatory_action: {entry.get('investigatory_action')}\n"
        f"trigger_conditions: {entry.get('trigger_conditions')}\n"
        f"applicable_gpu_models: {gpu_model_list}\n"
    ).format(n=examples_per_xid, gpu_models=gpu_model_list)


def extract_json(payload: str) -> dict[str, object]:
    payload = payload.strip()
    if payload.startswith("```"):
        payload = payload.strip("`")
        payload = payload.replace("json", "", 1).strip()
    return json.loads(payload)


def load_existing_counts(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not path.exists():
        return counts

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                counts[int(row.get("xid_code"))] += 1
            except (TypeError, ValueError):
                continue
    return counts


def run_generation(args: argparse.Namespace) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openai package is required; install gpu-health requirements first") from exc

    random.seed(args.seed)
    client = OpenAI()

    catalog_path = Path(args.catalog)
    rows = json.loads(catalog_path.read_text(encoding="utf-8"))
    if args.max_xids > 0:
        rows = rows[: args.max_xids]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_counts = load_existing_counts(out_path) if args.resume else Counter()
    file_mode = "a" if args.resume and out_path.exists() else "w"

    written = 0
    with out_path.open(file_mode, encoding="utf-8") as handle:
        for idx, entry in enumerate(rows, start=1):
            xid_code = int(entry.get("xid_code"))
            already_written = existing_counts.get(xid_code, 0)
            remaining_examples = max(args.examples_per_xid - already_written, 0)
            if remaining_examples == 0:
                print(f"[{idx}/{len(rows)}] xid={xid_code} skipped (already has {already_written})")
                continue

            gpu_models = applicable_gpu_models(entry)
            prompt = build_prompt(entry=entry, examples_per_xid=remaining_examples, gpu_models=gpu_models)

            examples: list[dict] = []
            last_error: Exception | None = None
            for attempt in range(1, args.max_attempts + 1):
                try:
                    response = client.chat.completions.create(
                        model=args.model,
                        temperature=0.7,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        timeout=args.request_timeout,
                    )
                    raw = response.choices[0].message.content or "{}"
                    parsed = extract_json(raw)
                    candidate_examples = parsed.get("examples", [])
                    if isinstance(candidate_examples, list):
                        examples = candidate_examples
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"[{idx}/{len(rows)}] xid={xid_code} attempt {attempt}/{args.max_attempts} failed: {exc}")
                    if attempt == args.max_attempts:
                        raise
                    time.sleep(min(2.0 * attempt, 10.0))
            if last_error is not None and not examples:
                raise last_error

            added_for_xid = 0
            for sample in examples:
                if added_for_xid >= remaining_examples:
                    break
                if not isinstance(sample, dict):
                    continue
                log_text = str(sample.get("log", "")).strip()
                if not log_text:
                    continue
                assistant = str(sample.get("assistant_response", "")).strip()
                log_source = str(sample.get("log_source", "unknown")).strip().lower()
                gpu_model = normalize_gpu_model(sample.get("gpu_model"), gpu_models)

                record = {
                    "xid_code": entry.get("xid_code"),
                    "mnemonic": entry.get("mnemonic"),
                    "description": entry.get("description"),
                    "severity": entry.get("severity"),
                    "immediate_action": entry.get("immediate_action"),
                    "investigatory_action": entry.get("investigatory_action"),
                    "trigger_conditions": entry.get("trigger_conditions"),
                    "gpu_model": gpu_model,
                    "log_source": log_source,
                    "log": log_text,
                    "assistant_response": assistant,
                    "messages": [
                        {"role": "system", "content": ASSISTANT_SYSTEM_PROMPT},
                        {"role": "user", "content": log_text},
                        {"role": "assistant", "content": assistant},
                    ],
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                written += 1
                added_for_xid += 1
                existing_counts[xid_code] += 1

            print(
                f"[{idx}/{len(rows)}] xid={xid_code} generated {added_for_xid} "
                f"(total {existing_counts[xid_code]}/{args.examples_per_xid})"
            )
            time.sleep(args.sleep_seconds)

    print(f"Wrote {written} examples to {out_path}")


def main() -> None:
    try:
        args = parse_args()
        run_generation(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
