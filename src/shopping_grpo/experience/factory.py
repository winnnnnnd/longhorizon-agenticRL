"""Build runtime and offline components from a secret-free experience config."""

from __future__ import annotations

from dataclasses import dataclass

from shopping_grpo.experience.compaction import (
    CONTEXT_COMPACTION_PROMPT_VERSION,
    DeepSeekFlashContextCompactor,
    SemanticCompactionCache,
)
from shopping_grpo.experience.config import (
    load_experience_config,
    public_experience_config,
    resolve_cache_path,
    resolve_project_path,
)
from shopping_grpo.experience.providers import (
    embedding_client_from_named_environment,
    json_client_from_named_environment,
)
from shopping_grpo.experience.retrieval import ExperienceRetriever
from shopping_grpo.experience.runtime import ExperienceRuntime
from shopping_grpo.experience.store import ExperienceStore


@dataclass(frozen=True)
class ExperienceComponents:
    runtime: ExperienceRuntime
    context_compactor: DeepSeekFlashContextCompactor
    public_config: dict


def build_experience_components(
    config_path,
    *,
    allow_validation_only: bool | None = None,
) -> ExperienceComponents:
    config = load_experience_config(config_path)
    store_config = config["store"]
    embeddings_path = resolve_project_path(config, store_config.get("embeddings"))
    store = ExperienceStore.load(
        cards_path=resolve_project_path(config, store_config["cards"]),
        manifest_path=resolve_project_path(config, store_config["manifest"]),
        embeddings_path=embeddings_path,
        allow_validation_only=(
            bool(store_config.get("allow_validation_only", False))
            if allow_validation_only is None
            else bool(allow_validation_only)
        ),
    )
    retrieval_config = config["retrieval"]
    backend = str(retrieval_config.get("backend", "lexical"))
    query_embedder = None
    if backend == "embedding":
        query_embedder = embedding_client_from_named_environment(
            retrieval_config["provider"]
        )
    retriever = ExperienceRetriever(
        store,
        top_k=int(retrieval_config.get("top_k", 3)),
        backend=backend,
        query_embedder=query_embedder,
        minimum_score=float(retrieval_config.get("minimum_score", 0.0)),
    )
    compaction_config = config["semantic_compaction"]
    compaction_provider = compaction_config["provider"]
    runtime = ExperienceRuntime(
        retriever,
        max_experience_tokens=int(config["injection"].get("max_tokens", 500)),
        semantic_compaction_manifest={
            "strategy": "deepseek_v4_flash_grounded",
            "prompt_version": CONTEXT_COMPACTION_PROMPT_VERSION,
            "model": compaction_provider["model"],
            "model_revision": compaction_provider["model_revision"],
            "provider_id": compaction_provider["provider_id"],
            "preserve_recent_groups": int(
                compaction_config.get("preserve_recent_groups", 2)
            ),
            "max_records_per_chunk": int(
                compaction_config.get("max_records_per_chunk", 12)
            ),
            "max_source_characters": int(
                compaction_config.get("max_source_characters", 60000)
            ),
            "cache_path_env": compaction_config.get("cache_path_env"),
            "cache_path": compaction_config.get("cache_path"),
        },
    )
    compaction_client = json_client_from_named_environment(
        compaction_config["provider"]
    )
    compactor = DeepSeekFlashContextCompactor(
        compaction_client,
        preserve_recent_groups=int(compaction_config.get("preserve_recent_groups", 2)),
        max_records_per_chunk=int(compaction_config.get("max_records_per_chunk", 12)),
        max_source_characters=int(compaction_config.get("max_source_characters", 60000)),
        cache=SemanticCompactionCache(resolve_cache_path(config)),
    )
    return ExperienceComponents(
        runtime=runtime,
        context_compactor=compactor,
        public_config=public_experience_config(config),
    )


def build_distillation_client(config_path):
    config = load_experience_config(config_path)
    return json_client_from_named_environment(config["distillation"]["provider"])
