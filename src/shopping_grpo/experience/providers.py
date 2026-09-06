"""OpenAI-compatible provider clients configured only through named environment variables."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import math
import os
from urllib.request import Request, urlopen

from shopping_grpo.evaluation.model_client import OpenAIJSONClient


class OpenAIEmbeddingClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        timeout: float = 120,
        dimensions: int | None = None,
        transport: Callable | None = None,
    ):
        if not model or not base_url or not api_key:
            raise ValueError("embedding model, base_url and api_key are required")
        self.model = str(model)
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key)
        self.timeout = float(timeout)
        self.dimensions = int(dimensions) if dimensions else None
        self.transport = transport

    def __call__(self, text: str) -> tuple[float, ...]:
        payload = {"model": self.model, "input": str(text), "encoding_format": "float"}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "shopping-agent-experience/0.1",
        }
        url = f"{self.base_url}/embeddings"
        if self.transport is not None:
            response = self.transport(url, payload, headers, self.timeout)
        else:
            request = Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urlopen(request, timeout=self.timeout) as raw:
                response = json.loads(raw.read().decode("utf-8"))
        data = response.get("data") if isinstance(response, Mapping) else None
        embedding = data[0].get("embedding") if isinstance(data, list) and data else None
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("embedding response is missing data[0].embedding")
        vector = tuple(float(item) for item in embedding)
        if any(not math.isfinite(item) for item in vector):
            raise ValueError("embedding response contains non-finite values")
        if self.dimensions is not None and len(vector) != self.dimensions:
            raise ValueError("embedding response dimension mismatch")
        return vector


def _provider_credentials(section: Mapping) -> tuple[str, str]:
    base_url_env = str(section.get("base_url_env") or "")
    api_key_env = str(section.get("api_key_env") or "")
    if not base_url_env or not api_key_env:
        raise ValueError("provider config must name base_url_env and api_key_env")
    base_url = os.environ.get(base_url_env)
    api_key = os.environ.get(api_key_env)
    if not base_url:
        raise ValueError(f"required provider environment variable is missing: {base_url_env}")
    if not api_key:
        raise ValueError(f"required provider environment variable is missing: {api_key_env}")
    return base_url, api_key


def json_client_from_named_environment(section: Mapping) -> OpenAIJSONClient:
    base_url, api_key = _provider_credentials(section)
    return OpenAIJSONClient(
        model=str(section.get("model") or "deepseek-v4-flash"),
        model_revision=str(
            section.get("model_revision")
            or section.get("model")
            or "deepseek-v4-flash"
        ),
        provider_id=str(section.get("provider_id") or "openai-compatible"),
        base_url=base_url,
        api_key=api_key,
        max_tokens=int(section.get("max_tokens", 4096)),
        timeout=float(section.get("timeout", 120)),
        retries=int(section.get("retries", 2)),
        retry_delay_seconds=float(section.get("retry_delay_seconds", 2)),
        response_format_json=True,
        thinking=bool(section.get("thinking", False)),
        reasoning_effort=str(section.get("reasoning_effort", "high")),
    )


def embedding_client_from_named_environment(section: Mapping) -> OpenAIEmbeddingClient:
    base_url, api_key = _provider_credentials(section)
    return OpenAIEmbeddingClient(
        model=str(section.get("model") or ""),
        base_url=base_url,
        api_key=api_key,
        timeout=float(section.get("timeout", 120)),
        dimensions=(int(section["dimensions"]) if section.get("dimensions") else None),
    )
