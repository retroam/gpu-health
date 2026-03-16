---
title: GPU Health Diagnostics
emoji: 🔥
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: false
---

# GPU Health Diagnostics

MVP implementation of the spec in `../gpu-health-spec.md`.

This repo provides a hybrid XID triage flow:
1. Deterministic extraction and catalog lookup
2. Tinker-backed LLM summarization with heuristic/OpenAI fallbacks
3. Data generation/splitting scripts for SFT
4. Training/eval entrypoints for Tinker-based LoRA

## Layout

```
gpu-health/
├── data/
│   ├── scrape_xid.py
│   ├── xid_catalog.json
│   ├── generate_synthetic.py
│   └── format_dataset.py
├── train/
│   └── train.py
├── eval/
│   └── eval.py
├── demo/
│   ├── cli.py
│   └── xid_lookup.py
└── requirements.txt
```

## Install

```bash
cd gpu-health
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Demo

```bash
cd gpu-health
python demo/cli.py "NVRM: Xid (PCI:0000:41:00): 79, pid=12345, GPU has fallen off the bus"
```

If `artifacts/gpu-health-v1.json` exists, `demo/cli.py` uses the saved Tinker runtime automatically.
Force specific modes with `--summary-mode heuristic|tinker|openai`.

JSON output:

```bash
python demo/cli.py --json "NVRM: Xid (PCI:0000:41:00): 79, pid=12345, GPU has fallen off the bus"
```

Optional OpenAI summary mode:

```bash
export OPENAI_API_KEY=...
python demo/cli.py --summary-mode openai "<log snippet>"
```

Explicit Tinker runtime:

```bash
python demo/cli.py \
  --summary-mode tinker \
  --tinker-manifest artifacts/gpu-health-v1.json \
  "<log snippet>"
```

Judge-ready Gradio app:

```bash
cd gpu-health
python demo/app.py
```

The Gradio app showcases three angles for judges:
- `Live Triage`: polished incident walkthrough on top of the real triage pipeline
- `CLI Showcase`: exact terminal command + captured CLI output
- `Project Story`: data, training, eval, and submission talking points

## Data Pipeline

### 1. Scrape NVIDIA XID catalog

```bash
cd gpu-health/data
python scrape_xid.py --out xid_catalog.json
```

### 2. Generate synthetic examples

```bash
export OPENAI_API_KEY=...
python generate_synthetic.py \
  --catalog xid_catalog.json \
  --out synthetic_logs.jsonl \
  --examples-per-xid 6
```

### 3. Build train/eval splits

```bash
python format_dataset.py \
  --input synthetic_logs.jsonl \
  --out-dir . \
  --train-xid-max 70 \
  --journalctl-train-frac 0.1
```

Produces:
- `train.jsonl`
- `eval_xid.jsonl`
- `eval_format.jsonl`

## Training (Tinker)

```bash
cd gpu-health/train
python train.py --train ../data/train.jsonl --epochs 3 --batch-size 8
```

Training writes a runtime manifest to `artifacts/gpu-health-v1.json` by default so the demo and eval scripts can load the saved sampler.

## Eval

`eval/eval.py` can either score an existing predictions file or generate predictions directly with the saved Tinker sampler.

```bash
cd gpu-health/eval
python eval.py \
  --dataset ../data/eval_xid.jsonl \
  --tinker-manifest ../artifacts/gpu-health-v1.json \
  --output-predictions predictions_xid.jsonl \
  --report eval_xid_report.json
```

To score a precomputed predictions file instead:

```bash
python eval.py \
  --dataset ../data/eval_xid.jsonl \
  --predictions predictions_xid.jsonl \
  --report eval_xid_report.json
```

## Latest Results

Artifacts generated on `2026-03-15` with the saved runtime in `artifacts/gpu-health-v1.json`.

- Dataset size: `642` synthetic examples across `107` XIDs (`6` examples per XID)
- Split sizes: `110` train, `486` held-out-XID eval, `46` held-out-format eval
- Training loss: epoch 1 `2.9895`, epoch 2 `1.1915`, epoch 3 `0.7471`
- Saved checkpoint: `tinker://dd1abd5e-14af-5db3-8e25-fb99ba240855:train:0/sampler_weights/gpu-health-v1`

Tinker evals below were run with `--max-tokens 64` against the saved runtime.

### Held-out XID Eval

- Report: `artifacts/eval_xid_report.json`
- Predictions: `artifacts/predictions_xid.jsonl`
- Rows: `486`
- XID exact accuracy: `68.72%` (`334/486`)
- Immediate action accuracy: `45.88%` (`223/486`)
- Investigatory action accuracy: `7.20%` (`35/486`)
- Both actions correct: `4.12%` (`20/486`)
- Summary BLEU: `0.2465`

### Held-out Format Eval

- Report: `artifacts/eval_format_report.json`
- Predictions: `artifacts/predictions_format.jsonl`
- Rows: `46`
- XID exact accuracy: `82.61%` (`38/46`)
- Immediate action accuracy: `41.30%` (`19/46`)
- Investigatory action accuracy: `13.04%` (`6/46`)
- Both actions correct: `4.35%` (`2/46`)
- Summary BLEU: `0.2457`

These results show that the model is learning to recover the XID code from unseen logs, but it is still weak on richer operator guidance, especially investigatory actions. The current model often produces terse outputs, so improving the supervised targets and prompting is the next highest-leverage step.

## Notes

- `data/xid_catalog.json` currently includes seed records so demo works immediately, but you should refresh it before training a real checkpoint.
- Re-run `data/scrape_xid.py` to refresh from NVIDIA source before production use.
- Tinker scripts require authenticated SDK access.
