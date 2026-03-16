"""Shared GPU XID triage helpers for CLI and Gradio demos."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lib.tinker_runtime import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    TinkerSampler,
    default_runtime_manifest_path,
    load_runtime_config,
)
from xid_lookup import build_heuristic_summary, extract_gpu_models, extract_pci_devices, extract_xid_code, load_catalog

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

DEFAULT_SYSTEM_PROMPT = (
    "You are a GPU cluster health diagnostic assistant. "
    "Given a raw NVIDIA log snippet, explain the XID context, keep recommendations "
    "grounded in the provided catalog fields, and avoid inventing unsupported causes."
)


def default_catalog_path(repo_root: Path) -> Path:
    return repo_root / "data" / "xid_catalog.json"


def default_tinker_manifest_path(repo_root: Path) -> Path:
    return default_runtime_manifest_path(repo_root, "gpu-health-v1")


def _openai_summary(log_text: str, catalog_context: str, model: str) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed; install dependencies first")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Catalog guidance:\n"
                    f"{catalog_context}\n\n"
                    "Raw log:\n"
                    f"{log_text}\n\n"
                    "Respond with concise operator guidance in plain text."
                ),
            },
        ],
    )
    return completion.choices[0].message.content or ""


def _build_tinker_messages(log_text: str, catalog_context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"{DEFAULT_SYSTEM_PROMPT}\n\n"
                "Deterministic catalog lookup:\n"
                f"{catalog_context}"
            ),
        },
        {"role": "user", "content": log_text},
    ]


def _tinker_summary(
    log_text: str,
    catalog_context: str,
    *,
    manifest_path: str | Path | None,
    model_path: str | None,
    base_model: str | None,
    max_tokens: int,
    temperature: float,
) -> str:
    config = load_runtime_config(
        manifest_path=manifest_path,
        model_path=model_path,
        base_model=base_model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    sampler = TinkerSampler(config)
    return sampler.sample_messages(_build_tinker_messages(log_text=log_text, catalog_context=catalog_context))


def _build_catalog_context(entry, pci_devices: list[str], gpu_models: list[str]) -> str:
    if entry is None:
        return "No matching XID entry found."

    return (
        f"xid_code: {entry.xid_code}\n"
        f"mnemonic: {entry.mnemonic}\n"
        f"description: {entry.description}\n"
        f"severity: {entry.severity}\n"
        f"immediate_action: {entry.immediate_action}\n"
        f"investigatory_action: {entry.investigatory_action}\n"
        f"trigger_conditions: {entry.trigger_conditions}\n"
        f"pci_devices: {', '.join(pci_devices) if pci_devices else 'unknown'}\n"
        f"gpu_models: {', '.join(gpu_models) if gpu_models else 'unknown'}"
    )


def diagnose_log(
    log_text: str,
    *,
    root: Path,
    catalog_path: str | Path | None = None,
    summary_mode: str = "auto",
    openai_model: str = "gpt-4o-mini",
    tinker_manifest: str | Path | None = None,
    tinker_model_path: str | None = None,
    tinker_base_model: str | None = None,
    tinker_max_tokens: int = DEFAULT_MAX_TOKENS,
    tinker_temperature: float = DEFAULT_TEMPERATURE,
) -> dict[str, Any]:
    catalog_file = Path(catalog_path) if catalog_path else default_catalog_path(root)
    manifest_file = Path(tinker_manifest) if tinker_manifest else default_tinker_manifest_path(root)

    xid_code = extract_xid_code(log_text)
    catalog = load_catalog(catalog_file)
    entry = catalog.get(xid_code) if xid_code is not None else None
    pci_devices = extract_pci_devices(log_text)
    gpu_models = extract_gpu_models(log_text)
    catalog_context = _build_catalog_context(entry=entry, pci_devices=pci_devices, gpu_models=gpu_models)

    resolved_summary_mode = summary_mode
    summary_warning: str | None = None
    has_tinker_config = bool(tinker_model_path or tinker_base_model or manifest_file.exists())
    if resolved_summary_mode == "auto":
        resolved_summary_mode = "tinker" if has_tinker_config else "heuristic"

    if resolved_summary_mode == "openai":
        summary = _openai_summary(log_text=log_text, catalog_context=catalog_context, model=openai_model)
    elif resolved_summary_mode == "tinker":
        try:
            summary = _tinker_summary(
                log_text=log_text,
                catalog_context=catalog_context,
                manifest_path=manifest_file if manifest_file.exists() else None,
                model_path=tinker_model_path,
                base_model=tinker_base_model,
                max_tokens=tinker_max_tokens,
                temperature=tinker_temperature,
            )
        except RuntimeError as exc:
            if summary_mode == "tinker":
                raise
            resolved_summary_mode = "heuristic"
            summary_warning = str(exc)
            summary = build_heuristic_summary(log_text=log_text, entry=entry, xid_code=xid_code)
    else:
        summary = build_heuristic_summary(log_text=log_text, entry=entry, xid_code=xid_code)

    payload: dict[str, Any] = {
        "xid_code": xid_code,
        "description": entry.description if entry else "Unknown",
        "severity": entry.severity if entry else "Unknown",
        "immediate_action": entry.immediate_action if entry else "UNKNOWN",
        "investigatory_action": entry.investigatory_action if entry else "UNKNOWN",
        "pci_devices": pci_devices,
        "gpu_models": gpu_models,
        "summary_mode": resolved_summary_mode,
        "summary": summary,
    }
    if summary_warning:
        payload["summary_warning"] = summary_warning
    return payload
