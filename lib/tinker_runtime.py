"""Helpers for saving and loading Tinker sampling runtimes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_RENDERER_NAME = "qwen3"
DEFAULT_MAX_TOKENS = 300
DEFAULT_TEMPERATURE = 0.0


@dataclass
class TinkerRuntimeConfig:
    base_model: str
    renderer_name: str = DEFAULT_RENDERER_NAME
    checkpoint_path: str | None = None
    save_name: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TinkerRuntimeConfig":
        return cls(
            base_model=str(data.get("base_model", "")),
            renderer_name=str(data.get("renderer_name", DEFAULT_RENDERER_NAME)),
            checkpoint_path=_coerce_tinker_path(data.get("checkpoint_path")),
            save_name=_coerce_optional_str(data.get("save_name")),
            max_tokens=int(data.get("max_tokens", DEFAULT_MAX_TOKENS)),
            temperature=float(data.get("temperature", DEFAULT_TEMPERATURE)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_tinker_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        return candidate or None
    candidate = getattr(value, "path", None)
    if isinstance(candidate, str):
        candidate = candidate.strip()
        return candidate or None
    if isinstance(value, dict):
        candidate = value.get("path")
        if isinstance(candidate, str):
            candidate = candidate.strip()
            return candidate or None
    return None


def default_runtime_manifest_path(repo_root: Path, save_name: str) -> Path:
    return repo_root / "artifacts" / f"{save_name}.json"


def ensure_tinker_imports():
    try:
        import tinker
        from tinker import types
        from tinker_cookbook import renderers
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing tinker dependencies. Install requirements and authenticate before using Tinker runtime."
        ) from exc

    return tinker, types, renderers


def load_runtime_config(
    *,
    manifest_path: str | Path | None = None,
    model_path: str | None = None,
    base_model: str | None = None,
    renderer_name: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> TinkerRuntimeConfig:
    data: dict[str, Any] = {}

    if manifest_path is not None:
        manifest = Path(manifest_path)
        if not manifest.exists():
            raise RuntimeError(f"Tinker manifest not found: {manifest}")
        data = json.loads(manifest.read_text(encoding="utf-8"))

    config = TinkerRuntimeConfig.from_dict(data)
    if base_model:
        config.base_model = base_model
    if model_path:
        config.checkpoint_path = model_path
    if renderer_name:
        config.renderer_name = renderer_name
    if max_tokens is not None:
        config.max_tokens = max_tokens
    if temperature is not None:
        config.temperature = temperature

    if not config.base_model and not config.checkpoint_path:
        raise RuntimeError("Provide a Tinker manifest, checkpoint path, or base model.")

    return config


def save_runtime_config(path: str | Path, config: TinkerRuntimeConfig) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(config.to_dict(), ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path


def export_sampling_runtime(service_client, training_client, *, base_model: str, save_name: str) -> tuple[object, TinkerRuntimeConfig]:
    checkpoint_path: str | None = None
    save_weights_for_sampler = getattr(training_client, "save_weights_for_sampler", None)

    if callable(save_weights_for_sampler):
        checkpoint_response = save_weights_for_sampler(name=save_name).result()
        checkpoint_path = _coerce_tinker_path(checkpoint_response)

    if checkpoint_path:
        sampling_client = service_client.create_sampling_client(model_path=checkpoint_path)
    else:
        sampling_client = training_client.save_weights_and_get_sampling_client(name=save_name)

    return sampling_client, TinkerRuntimeConfig(
        base_model=base_model,
        checkpoint_path=checkpoint_path,
        save_name=save_name,
    )


def load_sampling_tokenizer(sampling_client, config: TinkerRuntimeConfig):
    get_tokenizer = getattr(sampling_client, "get_tokenizer", None)
    if callable(get_tokenizer):
        return get_tokenizer()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("transformers is required to load a fallback tokenizer for Tinker sampling") from exc

    if not config.base_model:
        raise RuntimeError("base_model is required to load a fallback tokenizer for Tinker sampling")
    return AutoTokenizer.from_pretrained(config.base_model, fast=True)


class TinkerSampler:
    """Thin wrapper around Tinker sampling for chat-style prompts."""

    def __init__(self, config: TinkerRuntimeConfig):
        _, types, renderers = ensure_tinker_imports()
        self._types = types
        self._config = config

        import tinker

        self._service_client = tinker.ServiceClient()
        client_kwargs: dict[str, str] = {}
        if config.checkpoint_path:
            client_kwargs["model_path"] = config.checkpoint_path
        else:
            client_kwargs["base_model"] = config.base_model

        self._sampling_client = self._service_client.create_sampling_client(**client_kwargs)
        tokenizer = load_sampling_tokenizer(self._sampling_client, config)
        self._renderer = renderers.get_renderer(config.renderer_name, tokenizer)

    @property
    def config(self) -> TinkerRuntimeConfig:
        return self._config

    def sample_messages(self, messages: list[dict[str, str]], *, num_samples: int = 1) -> str:
        params = self._types.SamplingParams(
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            stop=self._renderer.get_stop_sequences(),
        )
        prompt = self._renderer.build_generation_prompt(messages)
        result = self._sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=num_samples).result()
        sequences = getattr(result, "sequences", None) or []
        if not sequences:
            return ""

        response_message, success = self._renderer.parse_response(sequences[0].tokens)
        if success and isinstance(response_message, dict):
            return str(response_message.get("content", "")).strip()

        content = getattr(response_message, "content", None)
        if isinstance(content, str):
            return content.strip()
        return str(response_message).strip()
