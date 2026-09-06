"""Configuration loading for the frozen-actor experience workflow."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import os
from pathlib import Path


EXPERIENCE_CONFIG_VERSION = "shopping-experience-config-v1"


def _find_project_root(source: Path) -> Path:
    resolved = source.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError("experience config is not located inside a project checkout")


def load_experience_config(path: str | Path) -> dict:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("experience config root must be an object")
    config = deepcopy(dict(payload))
    if config.get("schema_version") != EXPERIENCE_CONFIG_VERSION:
        raise ValueError("unsupported experience config schema_version")
    if config.get("enabled") is not True:
        raise ValueError("experience config must explicitly set enabled=true")
    for section in ("store", "retrieval", "injection", "semantic_compaction", "distillation"):
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"experience config section {section!r} is required")
    if config["semantic_compaction"].get("enabled") is not True:
        raise ValueError("experience injection requires semantic_compaction.enabled=true")
    compaction_provider = config["semantic_compaction"].get("provider")
    if not isinstance(compaction_provider, Mapping):
        raise ValueError("semantic_compaction.provider is required")
    model = str(compaction_provider.get("model") or "").casefold()
    if not model.startswith("deepseek-v4-flash"):
        raise ValueError("semantic compaction provider must use DeepSeek V4 Flash")
    if not str(compaction_provider.get("model_revision") or "").strip():
        raise ValueError("semantic compaction provider must pin model_revision")
    if not str(compaction_provider.get("provider_id") or "").strip():
        raise ValueError("semantic compaction provider must pin provider_id")
    if compaction_provider.get("thinking") is not False:
        raise ValueError("semantic compaction requires thinking=false")
    _reject_serialized_secrets(config)
    config["_config_path"] = str(source.resolve())
    config["_project_root"] = str(_find_project_root(source))
    return config


def resolve_project_path(config: Mapping, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def resolve_cache_path(config: Mapping) -> Path | None:
    section = config["semantic_compaction"]
    env_name = str(section.get("cache_path_env") or "")
    configured = os.environ.get(env_name) if env_name else None
    value = configured or section.get("cache_path")
    return resolve_project_path(config, value) if value else None


def public_experience_config(config: Mapping) -> dict:
    return {
        key: deepcopy(value)
        for key, value in config.items()
        if not str(key).startswith("_")
    }


def _reject_serialized_secrets(value: object, path: str = "experience_config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in {"api_key", "authorization", "token", "password", "secret"}:
                raise ValueError(f"{path}.{key} must name an environment variable, not a secret")
            _reject_serialized_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_serialized_secrets(child, f"{path}[{index}]")
