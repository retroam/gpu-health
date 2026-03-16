"""Judge-ready Gradio showcase for GPU Health Diagnostics."""

from __future__ import annotations

import html
import json
from functools import lru_cache
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover
    raise SystemExit("gradio is not installed. Run `pip install -r requirements.txt` first.") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.tinker_runtime import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from triage import default_tinker_manifest_path, diagnose_log

APP_CSS = """
:root {
  --bg-0: #07131f;
  --bg-1: rgba(8, 28, 43, 0.82);
  --bg-2: rgba(13, 38, 57, 0.95);
  --panel-border: rgba(255, 255, 255, 0.10);
  --ink: #f6efe3;
  --muted: #a8bfd1;
  --accent: #ff8a3d;
  --accent-2: #3cc6c6;
  --accent-3: #ffd166;
  --danger: #ff6b57;
  --shadow: 0 22px 70px rgba(0, 0, 0, 0.35);
}

.gradio-container {
  background:
    radial-gradient(circle at 0% 0%, rgba(255, 138, 61, 0.22), transparent 32%),
    radial-gradient(circle at 100% 10%, rgba(60, 198, 198, 0.15), transparent 30%),
    linear-gradient(180deg, #041019 0%, #091725 48%, #030910 100%);
  color: var(--ink);
  font-family: "Avenir Next", "Trebuchet MS", "Segoe UI", sans-serif;
}

.gradio-container .block {
  border: 1px solid var(--panel-border);
  background: linear-gradient(180deg, rgba(9, 26, 38, 0.94), rgba(7, 18, 29, 0.94));
  box-shadow: var(--shadow);
}

.gradio-container .tabs,
.gradio-container .tab-nav,
.gradio-container .tabitem {
  border: none !important;
}

.hero-shell,
.section-shell,
.artifact-shell {
  border: 1px solid var(--panel-border);
  background:
    linear-gradient(135deg, rgba(255, 138, 61, 0.07), transparent 36%),
    linear-gradient(180deg, rgba(7, 20, 31, 0.96), rgba(7, 17, 27, 0.92));
  border-radius: 28px;
  padding: 24px;
  position: relative;
  overflow: hidden;
  animation: rise 0.7s ease both;
}

.hero-shell::before,
.section-shell::before,
.artifact-shell::before {
  content: "";
  position: absolute;
  inset: auto -30% 70% auto;
  width: 280px;
  height: 280px;
  background: radial-gradient(circle, rgba(60, 198, 198, 0.16), transparent 65%);
  pointer-events: none;
}

.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border: 1px solid rgba(255, 209, 102, 0.35);
  border-radius: 999px;
  color: var(--accent-3);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 11px;
  margin-bottom: 14px;
}

.hero-shell h1,
.section-shell h2,
.artifact-shell h2 {
  margin: 0 0 12px;
  font-family: "Arial Rounded MT Bold", "Avenir Next", sans-serif;
  line-height: 1.02;
}

.hero-shell h1 {
  font-size: 46px;
  max-width: 10ch;
}

.hero-shell p,
.artifact-shell p,
.section-shell p,
.story-copy {
  color: var(--muted);
  font-size: 16px;
  line-height: 1.6;
}

.hero-grid,
.signal-grid,
.artifact-grid,
.flow-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 14px;
  margin-top: 18px;
}

.metric-card,
.signal-card,
.artifact-card,
.flow-card {
  border-radius: 20px;
  padding: 16px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: linear-gradient(180deg, rgba(16, 37, 53, 0.96), rgba(10, 26, 39, 0.96));
  min-height: 110px;
}

.metric-card strong,
.signal-card strong,
.artifact-card strong,
.flow-card strong {
  display: block;
  font-size: 14px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 8px;
}

.metric-card span,
.signal-value {
  display: block;
  font-size: 28px;
  font-weight: 700;
  color: var(--ink);
}

.signal-value.small {
  font-size: 18px;
  line-height: 1.4;
}

.accent-line {
  width: 72px;
  height: 4px;
  border-radius: 999px;
  margin: 18px 0;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
}

.pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.pill {
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: var(--muted);
  font-size: 12px;
}

.warning-box {
  border-left: 4px solid var(--danger);
  background: rgba(255, 107, 87, 0.08);
  border-radius: 16px;
  padding: 14px 16px;
  color: #ffd9d3;
  margin-top: 10px;
}

.command-box {
  border-radius: 18px;
  background: rgba(2, 10, 17, 0.9);
  border: 1px solid rgba(60, 198, 198, 0.25);
  padding: 16px;
  color: #d9efe7;
  font-family: "Menlo", "SFMono-Regular", monospace;
}

@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(14px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
"""

SUMMARY_MODE_NOTES = {
    "auto": "Auto mode prefers the trained Tinker runtime when it is available and falls back to heuristics otherwise.",
    "heuristic": "Deterministic mode is grounded entirely in the NVIDIA XID catalog and pattern extraction.",
    "tinker": "Tinker mode uses the trained runtime for richer context while keeping deterministic catalog guidance in the prompt.",
    "openai": "OpenAI mode uses the same grounded prompt structure but routes through the OpenAI API.",
}

EXAMPLE_PRIORITY = [
    ("Bus Drop / XID 79", 79),
    ("NVLink Fault / XID 74", 74),
    ("ECC Double-Bit / XID 48", 48),
    ("Memory Fault / XID 31", 31),
    ("Graphics Exception / XID 13", 13),
]

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def _safe_read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@lru_cache(maxsize=1)
def load_project_stats() -> dict[str, Any]:
    catalog_path = ROOT / "data" / "xid_catalog.json"
    synthetic_path = ROOT / "data" / "synthetic_logs.jsonl"
    catalog_rows = json.loads(catalog_path.read_text(encoding="utf-8"))
    synthetic_rows = _safe_read_jsonl(synthetic_path)
    unique_xids = {row["xid_code"] for row in synthetic_rows if "xid_code" in row}
    log_sources = sorted({row.get("log_source", "unknown") for row in synthetic_rows})
    return {
        "catalog_entries": len(catalog_rows),
        "synthetic_examples": len(synthetic_rows),
        "unique_xids": len(unique_xids),
        "log_sources": log_sources,
        "manifest_ready": default_tinker_manifest_path(ROOT).exists(),
    }


@lru_cache(maxsize=1)
def load_example_map() -> dict[str, str]:
    rows = _safe_read_jsonl(ROOT / "data" / "synthetic_logs.jsonl")
    example_map: dict[str, str] = {}
    for title, xid_code in EXAMPLE_PRIORITY:
        row = next((item for item in rows if item.get("xid_code") == xid_code), None)
        if row and row.get("log"):
            example_map[title] = str(row["log"])

    if example_map:
        return example_map

    return {
        "Bus Drop / XID 79": (
            "[2023-10-01 12:00:01] nvidia: GPU 0000:01:00.0: XID 79, GPU has fallen off the bus\n"
            "[2023-10-01 12:00:01] nvidia: GPU 0000:01:00.0: PCIe link failure\n"
            "[2023-10-01 12:00:02] nvidia: GPU 0000:01:00.0: Attempting to reset GPU"
        ),
    }


def render_hero() -> str:
    stats = load_project_stats()
    runtime_label = "Tinker runtime ready" if stats["manifest_ready"] else "Heuristic-first demo ready"
    log_sources = " + ".join(source.upper() for source in stats["log_sources"]) if stats["log_sources"] else "Dmesg + journalctl"
    return f"""
    <section class="hero-shell">
      <div class="eyebrow">SemiAnalysis x Fluidstack | Judge Demo</div>
      <h1>GPU Health Diagnostics</h1>
      <p>
        A polished front door for the existing CLI: deterministic NVIDIA XID triage,
        model-backed summarization, and a complete data-to-training story that judges can understand in under two minutes.
      </p>
      <div class="accent-line"></div>
      <div class="pill-row">
        <span class="pill">Hybrid deterministic + LLM reasoning</span>
        <span class="pill">Real operator workflow</span>
        <span class="pill">{html.escape(runtime_label)}</span>
        <span class="pill">{html.escape(log_sources)}</span>
      </div>
      <div class="hero-grid">
        <div class="metric-card">
          <strong>XID Catalog</strong>
          <span>{stats["catalog_entries"]}</span>
          NVIDIA playbooks loaded locally
        </div>
        <div class="metric-card">
          <strong>Synthetic Incidents</strong>
          <span>{stats["synthetic_examples"]}</span>
          Realistic training and demo logs
        </div>
        <div class="metric-card">
          <strong>XID Families</strong>
          <span>{stats["unique_xids"]}</span>
          Covered in the current dataset
        </div>
        <div class="metric-card">
          <strong>Demo Modes</strong>
          <span>4</span>
          Auto, heuristic, Tinker, OpenAI
        </div>
      </div>
    </section>
    """


def render_signal_cards(payload: dict[str, Any]) -> str:
    xid_code = payload.get("xid_code")
    xid_label = f"XID {xid_code}" if xid_code is not None else "No XID found"
    severity = payload.get("severity") or "Unspecified"
    pci_devices = ", ".join(payload.get("pci_devices") or []) or "None detected"
    gpu_models = ", ".join(payload.get("gpu_models") or []) or "None detected"
    return f"""
    <section class="section-shell">
      <h2>Operator Snapshot</h2>
      <div class="signal-grid">
        <div class="signal-card">
          <strong>Incident</strong>
          <span class="signal-value">{html.escape(xid_label)}</span>
          {html.escape(str(payload.get("description", "Unknown")))}
        </div>
        <div class="signal-card">
          <strong>Severity</strong>
          <span class="signal-value">{html.escape(str(severity))}</span>
          Catalog severity field
        </div>
        <div class="signal-card">
          <strong>Immediate Action</strong>
          <span class="signal-value small">{html.escape(str(payload.get("immediate_action", "UNKNOWN")))}</span>
        </div>
        <div class="signal-card">
          <strong>Investigatory Action</strong>
          <span class="signal-value small">{html.escape(str(payload.get("investigatory_action", "UNKNOWN")))}</span>
        </div>
        <div class="signal-card">
          <strong>Summary Mode</strong>
          <span class="signal-value">{html.escape(str(payload.get("summary_mode", "unknown")).upper())}</span>
          {html.escape(SUMMARY_MODE_NOTES.get(str(payload.get("summary_mode", "")), ""))}
        </div>
        <div class="signal-card">
          <strong>Detected Hardware</strong>
          <span class="signal-value small">{html.escape(gpu_models)}</span>
          GPUs
        </div>
        <div class="signal-card">
          <strong>PCI Devices</strong>
          <span class="signal-value small">{html.escape(pci_devices)}</span>
        </div>
      </div>
    </section>
    """


def render_warning(payload: dict[str, Any]) -> str:
    warning = payload.get("summary_warning")
    if not warning:
        return ""
    return f'<div class="warning-box"><strong>Fallback note:</strong> {html.escape(str(warning))}</div>'


def render_story() -> str:
    stats = load_project_stats()
    runtime_copy = (
        "The trained Tinker runtime is present, so judges can see the hybrid path live."
        if stats["manifest_ready"]
        else "No saved Tinker runtime is checked in, so the app demonstrates the deterministic path cleanly and still advertises the training hooks."
    )
    return f"""
    <section class="artifact-shell">
      <div class="eyebrow">What Else To Submit</div>
      <h2>More Than A UI Wrapper</h2>
      <p class="story-copy">
        The app is only the front door. Underneath it, the repo already contains a full pipeline:
        scrape the NVIDIA XID catalog, generate synthetic incidents, format training data,
        train a LoRA with Tinker, and score predictions with the eval harness.
      </p>
      <div class="artifact-grid">
        <div class="artifact-card">
          <strong>CLI</strong>
          <span class="signal-value small">`python demo/cli.py --json "&lt;log&gt;"`</span>
          A terminal-native operator path for SSH-first debugging.
        </div>
        <div class="artifact-card">
          <strong>Catalog Layer</strong>
          <span class="signal-value">{stats["catalog_entries"]}</span>
          Deterministic NVIDIA guidance is always available.
        </div>
        <div class="artifact-card">
          <strong>Training Corpus</strong>
          <span class="signal-value">{stats["synthetic_examples"]}</span>
          Synthetic incidents spanning {stats["unique_xids"]} XID families.
        </div>
        <div class="artifact-card">
          <strong>Hybrid Story</strong>
          <span class="signal-value small">{html.escape(runtime_copy)}</span>
        </div>
      </div>
      <div class="flow-grid">
        <div class="flow-card">
          <strong>1. Ingest</strong>
          Accept noisy logs from dmesg or journalctl.
        </div>
        <div class="flow-card">
          <strong>2. Ground</strong>
          Extract the XID, PCI IDs, and GPU models before any summarization.
        </div>
        <div class="flow-card">
          <strong>3. Explain</strong>
          Route through heuristics, Tinker, or OpenAI with the catalog embedded in context.
        </div>
        <div class="flow-card">
          <strong>4. Act</strong>
          Return immediate and investigatory actions operators can execute.
        </div>
      </div>
    </section>
    """


def story_markdown() -> str:
    return """
### Suggested submission flow

1. Lead with the **Live Triage** tab to show the product in one click.
2. Switch to **CLI Showcase** to prove there is a real operator-grade terminal workflow behind the UI.
3. End on **Project Story** to show this is a full-stack infrastructure project, not just a front-end demo.

### Strong repo assets already present

```bash
python data/scrape_xid.py --out xid_catalog.json
python data/generate_synthetic.py --catalog xid_catalog.json --out synthetic_logs.jsonl
python data/format_dataset.py --input synthetic_logs.jsonl --out-dir .
python train/train.py --train ../data/train.jsonl --epochs 3 --batch-size 8
python eval/eval.py --dataset ../data/eval_xid.jsonl --report eval_xid_report.json
```

### Judge-facing pitch

- Deterministic XID extraction gives grounded operator actions immediately.
- The model layer adds readable incident context without losing the catalog constraints.
- The repo includes data generation, training, inference, and evaluation entrypoints, so the project reads like a real product pipeline.
"""


def build_cli_command(
    log_text: str,
    summary_mode: str,
    output_format: str,
    openai_model: str,
    tinker_manifest: str,
    tinker_model_path: str,
    tinker_base_model: str,
    tinker_max_tokens: int,
    tinker_temperature: float,
) -> str:
    command = ["python", "demo/cli.py", "--summary-mode", summary_mode]
    default_manifest = str(default_tinker_manifest_path(ROOT))
    if output_format == "json":
        command.append("--json")
    if summary_mode == "openai" and openai_model and openai_model != DEFAULT_OPENAI_MODEL:
        command.extend(["--openai-model", openai_model])
    if tinker_manifest and tinker_manifest != default_manifest:
        command.extend(["--tinker-manifest", tinker_manifest])
    if tinker_model_path:
        command.extend(["--tinker-model-path", tinker_model_path])
    if tinker_base_model:
        command.extend(["--tinker-base-model", tinker_base_model])
    if tinker_max_tokens != DEFAULT_MAX_TOKENS:
        command.extend(["--tinker-max-tokens", str(tinker_max_tokens)])
    if abs(tinker_temperature - DEFAULT_TEMPERATURE) > 1e-9:
        command.extend(["--tinker-temperature", str(tinker_temperature)])
    if log_text.strip():
        command.append(log_text.strip())
    return " ".join(shlex.quote(part) for part in command)


def read_uploaded_log(file_path: str | None) -> str:
    if not file_path:
        return ""
    return Path(file_path).read_text(encoding="utf-8")


def load_example(example_name: str) -> str:
    return load_example_map().get(example_name, "")


def run_triage(
    log_text: str,
    summary_mode: str,
    openai_model: str,
    tinker_manifest: str,
    tinker_model_path: str,
    tinker_base_model: str,
    tinker_max_tokens: int,
    tinker_temperature: float,
) -> tuple[str, str, str, dict[str, Any], str]:
    if not log_text.strip():
        empty_payload = {"error": "Paste a GPU log snippet or load an example first."}
        return (
            render_signal_cards(
                {
                    "xid_code": None,
                    "description": "Awaiting input",
                    "severity": "Unknown",
                    "immediate_action": "N/A",
                    "investigatory_action": "N/A",
                    "summary_mode": summary_mode,
                    "gpu_models": [],
                    "pci_devices": [],
                }
            ),
            "### Paste a log snippet\nThe app will extract the XID and operator actions once it has input.",
            "",
            empty_payload,
            "",
        )

    command = build_cli_command(
        log_text=log_text,
        summary_mode=summary_mode,
        output_format="json",
        openai_model=openai_model,
        tinker_manifest=tinker_manifest,
        tinker_model_path=tinker_model_path,
        tinker_base_model=tinker_base_model,
        tinker_max_tokens=tinker_max_tokens,
        tinker_temperature=tinker_temperature,
    )
    try:
        payload = diagnose_log(
            log_text=log_text,
            root=ROOT,
            summary_mode=summary_mode,
            openai_model=openai_model,
            tinker_manifest=tinker_manifest or None,
            tinker_model_path=tinker_model_path or None,
            tinker_base_model=tinker_base_model or None,
            tinker_max_tokens=tinker_max_tokens,
            tinker_temperature=tinker_temperature,
        )
    except RuntimeError as exc:
        payload = {"error": str(exc), "summary_mode": summary_mode}
        summary = f"### Runtime error\n`{exc}`"
        warning = f'<div class="warning-box"><strong>Runtime error:</strong> {html.escape(str(exc))}</div>'
        return render_signal_cards(
            {
                "xid_code": None,
                "description": "Inference path unavailable",
                "severity": "Unknown",
                "immediate_action": "Check runtime configuration",
                "investigatory_action": "Switch to heuristic mode or supply credentials",
                "summary_mode": summary_mode,
                "gpu_models": [],
                "pci_devices": [],
            }
        ), summary, warning, payload, command

    summary_md = f"### Model Readout\n{payload['summary']}"
    return render_signal_cards(payload), summary_md, render_warning(payload), payload, command


def run_cli_showcase(
    log_text: str,
    summary_mode: str,
    output_format: str,
    openai_model: str,
    tinker_manifest: str,
    tinker_model_path: str,
    tinker_base_model: str,
    tinker_max_tokens: int,
    tinker_temperature: float,
) -> tuple[str, str]:
    command_text = build_cli_command(
        log_text=log_text,
        summary_mode=summary_mode,
        output_format=output_format,
        openai_model=openai_model,
        tinker_manifest=tinker_manifest,
        tinker_model_path=tinker_model_path,
        tinker_base_model=tinker_base_model,
        tinker_max_tokens=tinker_max_tokens,
        tinker_temperature=tinker_temperature,
    )
    if not log_text.strip():
        return command_text, "Paste a log snippet or load an example first."

    command = [sys.executable, str(ROOT / "demo" / "cli.py"), "--summary-mode", summary_mode]
    default_manifest = str(default_tinker_manifest_path(ROOT))
    if output_format == "json":
        command.append("--json")
    if summary_mode == "openai" and openai_model and openai_model != DEFAULT_OPENAI_MODEL:
        command.extend(["--openai-model", openai_model])
    if tinker_manifest and tinker_manifest != default_manifest:
        command.extend(["--tinker-manifest", tinker_manifest])
    if tinker_model_path:
        command.extend(["--tinker-model-path", tinker_model_path])
    if tinker_base_model:
        command.extend(["--tinker-base-model", tinker_base_model])
    if tinker_max_tokens != DEFAULT_MAX_TOKENS:
        command.extend(["--tinker-max-tokens", str(tinker_max_tokens)])
    if abs(tinker_temperature - DEFAULT_TEMPERATURE) > 1e-9:
        command.extend(["--tinker-temperature", str(tinker_temperature)])
    command.append(log_text.strip())

    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return command_text, "CLI invocation timed out after 45 seconds."

    output = result.stdout.strip()
    if result.stderr.strip():
        output = f"{output}\n\n[stderr]\n{result.stderr.strip()}".strip()
    if not output:
        output = f"CLI exited with code {result.returncode} and produced no output."
    return command_text, output


def build_app() -> gr.Blocks:
    example_names = list(load_example_map())
    default_example = example_names[0] if example_names else None
    default_log = load_example(default_example) if default_example else ""
    default_manifest = str(default_tinker_manifest_path(ROOT))

    with gr.Blocks(css=APP_CSS, title="GPU Health Diagnostics Showcase") as demo:
        gr.HTML(render_hero())

        with gr.Tabs():
            with gr.Tab("Live Triage"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown(
                            "Paste a raw NVIDIA incident, load a curated demo case, or upload a log file. "
                            "This tab uses the same triage core as the CLI."
                        )
                        example_picker = gr.Dropdown(
                            choices=example_names,
                            value=default_example,
                            label="Curated demo scenario",
                            allow_custom_value=False,
                        )
                        log_input = gr.Textbox(
                            label="Raw GPU log snippet",
                            value=default_log,
                            lines=14,
                            max_lines=22,
                            placeholder="Paste a dmesg or journalctl snippet here...",
                        )
                        log_file = gr.File(label="Optional log file", type="filepath", file_types=[".txt", ".log"])
                        summary_mode = gr.Radio(
                            choices=["auto", "heuristic", "tinker", "openai"],
                            value="auto",
                            label="Summary mode",
                        )
                        with gr.Accordion("Advanced inference settings", open=False):
                            openai_model = gr.Textbox(label="OpenAI model", value=DEFAULT_OPENAI_MODEL)
                            tinker_manifest = gr.Textbox(label="Tinker manifest", value=default_manifest)
                            tinker_model_path = gr.Textbox(label="Explicit Tinker checkpoint", value="")
                            tinker_base_model = gr.Textbox(label="Fallback base model", value="")
                            tinker_max_tokens = gr.Slider(
                                minimum=64,
                                maximum=800,
                                value=DEFAULT_MAX_TOKENS,
                                step=1,
                                label="Tinker max tokens",
                            )
                            tinker_temperature = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=DEFAULT_TEMPERATURE,
                                step=0.05,
                                label="Tinker temperature",
                            )
                        analyze_btn = gr.Button("Analyze Incident", variant="primary")

                    with gr.Column(scale=6):
                        signal_cards = gr.HTML(render_signal_cards({}))
                        summary_md = gr.Markdown()
                        warning_html = gr.HTML()
                        payload_json = gr.JSON(label="Structured triage payload")
                        command_preview = gr.Code(
                            label="Equivalent CLI command",
                            language="bash",
                            value="python demo/cli.py --summary-mode auto --json '<log snippet>'",
                        )

                example_picker.change(load_example, inputs=example_picker, outputs=log_input)
                log_file.upload(read_uploaded_log, inputs=log_file, outputs=log_input)
                analyze_btn.click(
                    run_triage,
                    inputs=[
                        log_input,
                        summary_mode,
                        openai_model,
                        tinker_manifest,
                        tinker_model_path,
                        tinker_base_model,
                        tinker_max_tokens,
                        tinker_temperature,
                    ],
                    outputs=[signal_cards, summary_md, warning_html, payload_json, command_preview],
                )

            with gr.Tab("CLI Showcase"):
                gr.HTML(
                    """
                    <section class="section-shell">
                      <h2>Exact Operator Path</h2>
                      <p>
                        Judges often see polished demos that hide the real workflow. This tab proves the opposite:
                        the web UI is sitting on top of an actual terminal-ready CLI that can run over SSH,
                        emit JSON, and slot into larger GPU fleet tooling.
                      </p>
                    </section>
                    """
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        cli_example_picker = gr.Dropdown(
                            choices=example_names,
                            value=default_example,
                            label="Scenario",
                            allow_custom_value=False,
                        )
                        cli_log_input = gr.Textbox(label="CLI input log", value=default_log, lines=12, max_lines=18)
                        cli_summary_mode = gr.Radio(
                            choices=["auto", "heuristic", "tinker", "openai"],
                            value="auto",
                            label="Summary mode",
                        )
                        cli_output_format = gr.Radio(choices=["json", "plain"], value="json", label="CLI output format")
                        cli_run_btn = gr.Button("Run CLI", variant="primary")
                    with gr.Column(scale=6):
                        cli_command = gr.Code(label="Pasteable command", language="bash")
                        cli_output = gr.Code(label="Captured CLI output", language="shell")

                cli_example_picker.change(load_example, inputs=cli_example_picker, outputs=cli_log_input)
                cli_run_btn.click(
                    run_cli_showcase,
                    inputs=[
                        cli_log_input,
                        cli_summary_mode,
                        cli_output_format,
                        openai_model,
                        tinker_manifest,
                        tinker_model_path,
                        tinker_base_model,
                        tinker_max_tokens,
                        tinker_temperature,
                    ],
                    outputs=[cli_command, cli_output],
                )

            with gr.Tab("Project Story"):
                gr.HTML(render_story())
                gr.Markdown(story_markdown())

    return demo


demo = build_app()


if __name__ == "__main__":
    demo.launch()
