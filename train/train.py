"""LoRA SFT training loop using Tinker for GPU XID diagnostics."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.tinker_runtime import default_runtime_manifest_path, ensure_tinker_imports, export_sampling_runtime, save_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA model on XID SFT dataset")
    parser.add_argument("--train", default="../data/train.jsonl", help="Path to train JSONL")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-name", default="gpu-health-v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Optional path to write Tinker runtime manifest for demo/eval inference",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def messages_to_datum(messages: list[dict], renderer, types):
    model_input, weights = renderer.build_supervised_example(messages)
    if hasattr(model_input, "tolist"):
        tokens = model_input.tolist()
    else:
        tokens = model_input.to_ints()
    if hasattr(weights, "tolist"):
        token_weights = weights.tolist()
    else:
        token_weights = weights.to_ints()
    if len(tokens) < 2:
        raise ValueError("Message rendering produced fewer than 2 tokens")

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs={
            "weights": token_weights[1:],
            "target_tokens": tokens[1:],
        },
    )


def to_numpy_array(value) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        return np.asarray(value.to_numpy(), dtype=float)
    if hasattr(value, "tolist"):
        return np.asarray(value.tolist(), dtype=float)
    return np.asarray(value, dtype=float)


def run_training(args: argparse.Namespace) -> None:
    tinker, types, renderers = ensure_tinker_imports()

    random.seed(args.seed)
    np.random.seed(args.seed)

    rows = load_jsonl(Path(args.train))
    if not rows:
        raise RuntimeError("Training dataset is empty")

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=args.base_model)
    tokenizer = training_client.get_tokenizer()
    renderer = renderers.get_renderer("qwen3", tokenizer)

    dataset = [messages_to_datum(row["messages"], renderer, types) for row in rows]

    for epoch in range(args.epochs):
        random.shuffle(dataset)
        epoch_losses: list[float] = []

        for batch_start in range(0, len(dataset), args.batch_size):
            batch = dataset[batch_start : batch_start + args.batch_size]

            fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(types.AdamParams(learning_rate=args.learning_rate))

            fwdbwd_result = fwdbwd_future.result()
            optim_future.result()

            logprobs = np.concatenate(
                [to_numpy_array(loss_output["logprobs"]) for loss_output in fwdbwd_result.loss_fn_outputs]
            )
            weights = np.concatenate(
                [to_numpy_array(datum.loss_fn_inputs["weights"]) for datum in batch]
            )
            if weights.sum() == 0:
                continue
            loss = float(-np.dot(logprobs, weights) / weights.sum())
            epoch_losses.append(loss)

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(f"epoch={epoch + 1}/{args.epochs} avg_loss={avg_loss:.4f}")

    sampling_client, runtime_config = export_sampling_runtime(
        service_client,
        training_client,
        base_model=args.base_model,
        save_name=args.save_name,
    )
    manifest_path = Path(args.manifest_out) if args.manifest_out else default_runtime_manifest_path(ROOT, args.save_name)
    save_runtime_config(manifest_path, runtime_config)

    print("Saved checkpoint and created sampling client")
    print(f"runtime_manifest={manifest_path}")
    if runtime_config.checkpoint_path:
        print(f"checkpoint_path={runtime_config.checkpoint_path}")
    else:
        print("checkpoint_path=<ephemeral sampling client only>")
    print(f"sampling_client={sampling_client}")


def main() -> None:
    try:
        args = parse_args()
        run_training(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
