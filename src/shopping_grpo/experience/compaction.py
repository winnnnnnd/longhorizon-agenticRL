"""Grounded DeepSeek V4 Flash compaction for experience-augmented actor context."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
import re

from shopping_grpo.environment.context import (
    ContextBudgetError,
    flatten_chat_tool_groups,
    split_chat_tool_groups,
)
from shopping_grpo.environment.product_id import PRODUCT_ID_CAPTURE
from shopping_grpo.evaluation.artifacts import append_jsonl_fsync, iter_jsonl
from shopping_grpo.experience.contracts import canonical_json, canonical_sha256


CONTEXT_SUMMARY_VERSION = "shopping-context-summary-v1"
CONTEXT_COMPACTION_PROMPT_VERSION = "shopping-context-compaction-v1"
SUMMARY_KINDS = frozenset(
    {"search", "candidate", "option", "verification", "rejection", "decision", "unresolved"}
)
_PRODUCT_ID = re.compile(rf"(?<!\d)({PRODUCT_ID_CAPTURE})(?!\d)")
_FORBIDDEN = ("target_asin", "gold_purchase", "reward_detail", "audit_only_raw_observation")


SYSTEM_PROMPT = f"""你是 Shopping Agent 的上下文压缩器，只压缩已经结束的旧交互。
你会收到带稳定 event_id 的 Actor 可见历史。固定 system prompt、原始用户 query、
当前经验块和最近交互不会交给你，也绝不能由你改写。

只输出 JSON，schema_version 必须是 {CONTEXT_SUMMARY_VERSION}。source_event_ids 必须
按输入顺序逐项原样返回。records 最多达到输入 max_records，每项 kind 只能取：
search、candidate、option、verification、rejection、decision、unresolved。

每项 fact 必须是未来决策仍需要的简短事实，不能添加商品事实或推断。每项至少提供
一个 evidence，event_id 必须来自输入，quote 必须是该 event source_text 中逐字连续
出现的短文本。不要输出 Reward、Gold、隐藏目标、总分、建议或新的行动指令。
无法找到逐字证据的内容不要保留。不要输出 Markdown 或额外字段。"""


class SemanticCompactionError(ContextBudgetError):
    """Fail closed when a grounded summary cannot preserve the context contract."""


class SemanticCompactionCache:
    """Append-only cache keyed by model, prompt version and actor-visible source hash."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._rows = {}
        if self.path is not None and self.path.exists():
            for row in iter_jsonl(self.path):
                key = str(row.get("cache_key") or "")
                if not key:
                    raise SemanticCompactionError("semantic compaction cache row has no key")
                if key in self._rows and self._rows[key] != row:
                    raise SemanticCompactionError("semantic compaction cache has conflicting rows")
                self._rows[key] = row

    def get(self, key: str) -> dict | None:
        row = self._rows.get(str(key))
        return deepcopy(row) if row is not None else None

    def put(self, row: Mapping) -> None:
        value = deepcopy(dict(row))
        key = str(value.get("cache_key") or "")
        if not key:
            raise SemanticCompactionError("semantic compaction cache write has no key")
        existing = self._rows.get(key)
        if existing is not None:
            if existing != value:
                raise SemanticCompactionError("semantic compaction cache key changed result")
            return
        if self.path is not None:
            append_jsonl_fsync(self.path, value)
        self._rows[key] = value


def _tool_call_view(tool_call: object) -> dict:
    if not isinstance(tool_call, Mapping):
        return {"id": None, "name": "", "arguments": {}}
    function = tool_call.get("function")
    function = function if isinstance(function, Mapping) else {}
    raw_arguments = function.get("arguments") or {}
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {}
    else:
        arguments = deepcopy(dict(raw_arguments)) if isinstance(raw_arguments, Mapping) else {}
    return {
        "id": tool_call.get("id"),
        "name": str(function.get("name") or ""),
        "arguments": arguments,
    }


def _source_events(groups: list[list[Mapping]]) -> list[dict]:
    events = []
    for group in groups:
        assistant = group[0]
        calls = {
            str(call.get("id")): _tool_call_view(call)
            for call in assistant.get("tool_calls") or []
            if isinstance(call, Mapping) and call.get("id") is not None
        }
        for tool_message in group[1:]:
            call_id = str(tool_message.get("tool_call_id") or "")
            payload = {
                "assistant_text": str(assistant.get("content") or ""),
                "tool_call": calls.get(call_id, {"id": call_id, "name": "", "arguments": {}}),
                "tool_name": str(tool_message.get("name") or ""),
                "tool_observation": str(tool_message.get("content") or ""),
            }
            events.append(
                {
                    "event_id": f"h{len(events) + 1:04d}",
                    "source_text": canonical_json(payload),
                }
            )
    return events


def _chunks(events: list[dict], maximum_characters: int) -> list[list[dict]]:
    chunks = []
    current = []
    size = 0
    for event in events:
        event_size = len(event["source_text"]) + len(event["event_id"]) + 32
        if event_size > maximum_characters:
            raise SemanticCompactionError(
                "one actor-visible history event exceeds the compaction source budget"
            )
        if current and size + event_size > maximum_characters:
            chunks.append(current)
            current = []
            size = 0
        current.append(event)
        size += event_size
    if current:
        chunks.append(current)
    return chunks


def _validate_summary(result: object, source_events: list[dict], max_records: int) -> dict:
    if not isinstance(result, Mapping):
        raise SemanticCompactionError("semantic context summary must be an object")
    result = deepcopy(dict(result))
    if result.get("schema_version") != CONTEXT_SUMMARY_VERSION:
        raise SemanticCompactionError("semantic context summary has an unsupported schema")
    expected_ids = [event["event_id"] for event in source_events]
    if result.get("source_event_ids") != expected_ids:
        raise SemanticCompactionError("semantic context summary source_event_ids differ from input")
    records = result.get("records")
    if not isinstance(records, list) or not records:
        raise SemanticCompactionError("semantic context summary records cannot be empty")
    if len(records) > int(max_records):
        raise SemanticCompactionError("semantic context summary contains too many records")
    source_by_id = {event["event_id"]: event["source_text"] for event in source_events}
    source_product_ids = set(_PRODUCT_ID.findall("\n".join(source_by_id.values())))
    normalized_records = []
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, Mapping):
            raise SemanticCompactionError(f"summary record {index} must be an object")
        record = dict(raw_record)
        if set(record) != {"kind", "fact", "evidence"}:
            raise SemanticCompactionError(f"summary record {index} has unexpected fields")
        if record["kind"] not in SUMMARY_KINDS:
            raise SemanticCompactionError(f"summary record {index} has an unknown kind")
        if not isinstance(record["fact"], str) or not record["fact"].strip():
            raise SemanticCompactionError(f"summary record {index} fact is empty")
        if len(record["fact"]) > 500:
            raise SemanticCompactionError(f"summary record {index} fact is too long")
        fact_ids = set(_PRODUCT_ID.findall(record["fact"]))
        if not fact_ids <= source_product_ids:
            raise SemanticCompactionError(f"summary record {index} invented a product ID")
        if any(term in record["fact"].casefold() for term in _FORBIDDEN):
            raise SemanticCompactionError(f"summary record {index} contains hidden result data")
        evidence = record["evidence"]
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 3:
            raise SemanticCompactionError(f"summary record {index} evidence count is invalid")
        normalized_evidence = []
        for evidence_index, raw_evidence in enumerate(evidence):
            if not isinstance(raw_evidence, Mapping) or set(raw_evidence) != {"event_id", "quote"}:
                raise SemanticCompactionError(
                    f"summary record {index} evidence {evidence_index} is malformed"
                )
            event_id = str(raw_evidence["event_id"])
            quote = str(raw_evidence["quote"])
            if event_id not in source_by_id:
                raise SemanticCompactionError(
                    f"summary record {index} references an unknown event"
                )
            if not quote or len(quote) > 240 or quote not in source_by_id[event_id]:
                raise SemanticCompactionError(
                    f"summary record {index} evidence quote is not grounded"
                )
            normalized_evidence.append({"event_id": event_id, "quote": quote})
        normalized_records.append(
            {
                "kind": record["kind"],
                "fact": record["fact"].strip(),
                "evidence": normalized_evidence,
            }
        )
    result["records"] = normalized_records
    return result


def _render_summary(records: list[Mapping], source_hash: str) -> str:
    lines = [
        f"[GROUNDED_HISTORY_SUMMARY_V1 source_hash={source_hash}]",
        "以下仅是已结束旧交互的证据化摘要；当前页面和最新工具结果优先。",
    ]
    for record in records:
        event_ids = ",".join(item["event_id"] for item in record["evidence"])
        lines.append(f"- {record['kind']} [{event_ids}]: {record['fact']}")
    lines.append("[/GROUNDED_HISTORY_SUMMARY_V1]")
    return "\n".join(lines)


def _append_to_system(anchor: list[Mapping], summary: str) -> list[dict]:
    result = [deepcopy(dict(message)) for message in anchor]
    system_index = next(
        (index for index, message in enumerate(result) if message.get("role") == "system"),
        None,
    )
    if system_index is None:
        raise SemanticCompactionError("semantic compaction requires a fixed system message")
    original = str(result[system_index].get("content") or "").rstrip()
    result[system_index]["content"] = original + "\n\n" + summary
    return result


class DeepSeekFlashContextCompactor:
    """Summarize only old complete groups and preserve anchors/recent groups verbatim."""

    def __init__(
        self,
        client,
        *,
        preserve_recent_groups: int = 2,
        max_records_per_chunk: int = 12,
        max_source_characters: int = 60000,
        cache: SemanticCompactionCache | None = None,
    ):
        if int(preserve_recent_groups) < 1:
            raise ValueError("preserve_recent_groups must be positive")
        if int(max_records_per_chunk) < 1:
            raise ValueError("max_records_per_chunk must be positive")
        if int(max_source_characters) < 4096:
            raise ValueError("max_source_characters must be at least 4096")
        self.client = client
        self.preserve_recent_groups = int(preserve_recent_groups)
        self.max_records_per_chunk = int(max_records_per_chunk)
        self.max_source_characters = int(max_source_characters)
        self.cache = cache or SemanticCompactionCache(None)

    def _summarize_chunk(self, source_events: list[dict]) -> tuple[dict, dict, bool]:
        source_hash = canonical_sha256(source_events)
        cache_key = canonical_sha256(
            {
                "model": self.client.model,
                "model_revision": self.client.model_revision,
                "provider_id": self.client.provider_id,
                "prompt_version": CONTEXT_COMPACTION_PROMPT_VERSION,
                "source_hash": source_hash,
                "max_records": self.max_records_per_chunk,
            }
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            expected_cache_fields = {
                "schema_version": "shopping-context-compaction-cache-v1",
                "cache_key": cache_key,
                "model": self.client.model,
                "model_revision": self.client.model_revision,
                "provider_id": self.client.provider_id,
                "prompt_version": CONTEXT_COMPACTION_PROMPT_VERSION,
                "source_hash": source_hash,
            }
            if any(
                cached.get(field) != expected
                for field, expected in expected_cache_fields.items()
            ):
                raise SemanticCompactionError(
                    "semantic compaction cache metadata does not match its key"
                )
            result = _validate_summary(
                cached.get("result"), source_events, self.max_records_per_chunk
            )
            if cached.get("result_hash") != canonical_sha256(result):
                raise SemanticCompactionError(
                    "semantic compaction cache result hash mismatch"
                )
            return result, deepcopy(cached.get("provider_metadata") or {}), True
        request = {
            "schema_version": "shopping-context-compaction-input-v1",
            "prompt_version": CONTEXT_COMPACTION_PROMPT_VERSION,
            "max_records": self.max_records_per_chunk,
            "source_events": deepcopy(source_events),
        }
        response = self.client.complete_json(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(request, ensure_ascii=False, sort_keys=True),
                },
            ]
        )
        result = _validate_summary(
            response.get("result"), source_events, self.max_records_per_chunk
        )
        row = {
            "schema_version": "shopping-context-compaction-cache-v1",
            "cache_key": cache_key,
            "model": self.client.model,
            "model_revision": self.client.model_revision,
            "provider_id": self.client.provider_id,
            "prompt_version": CONTEXT_COMPACTION_PROMPT_VERSION,
            "source_hash": source_hash,
            "result_hash": canonical_sha256(result),
            "result": result,
            "provider_metadata": deepcopy(response.get("metadata") or {}),
        }
        self.cache.put(row)
        return result, row["provider_metadata"], False

    def compact(self, messages, tools, *, count_tokens, max_input_tokens):
        original = [deepcopy(dict(message)) for message in messages]
        original_tokens = int(count_tokens(original, tools))
        if original_tokens <= int(max_input_tokens):
            return original, {
                "strategy": "deepseek_v4_flash_grounded",
                "original_tokens": original_tokens,
                "final_tokens": original_tokens,
                "removed_tokens": 0,
                "removed_groups": 0,
                "removed_messages": 0,
                "cache_hits": 0,
            }
        anchor, groups = split_chat_tool_groups(original)
        if len(groups) <= self.preserve_recent_groups:
            raise SemanticCompactionError(
                "fixed prompt plus protected recent tool groups exceed the actor input budget"
            )
        old_groups = groups[: -self.preserve_recent_groups]
        recent_groups = groups[-self.preserve_recent_groups :]
        source_events = _source_events(old_groups)
        if not source_events:
            raise SemanticCompactionError("no complete old events are available for compression")
        results = []
        metadata = []
        cache_hits = 0
        for chunk in _chunks(source_events, self.max_source_characters):
            result, provider_metadata, cache_hit = self._summarize_chunk(chunk)
            results.append(result)
            metadata.append(provider_metadata)
            cache_hits += int(cache_hit)
        records = []
        seen_records = set()
        for result in results:
            for record in result["records"]:
                key = canonical_sha256(record)
                if key not in seen_records:
                    records.append(record)
                    seen_records.add(key)
        source_hash = canonical_sha256(source_events)
        summary = _render_summary(records, source_hash)
        compacted = _append_to_system(anchor, summary) + flatten_chat_tool_groups(recent_groups)
        final_tokens = int(count_tokens(compacted, tools))
        if final_tokens > int(max_input_tokens):
            raise SemanticCompactionError(
                "grounded summary plus protected context exceeds the actor input budget"
            )
        return compacted, {
            "strategy": "deepseek_v4_flash_grounded",
            "model": self.client.model,
            "model_revision": self.client.model_revision,
            "provider_id": self.client.provider_id,
            "prompt_version": CONTEXT_COMPACTION_PROMPT_VERSION,
            "source_hash": source_hash,
            "summary_hash": canonical_sha256(summary),
            "source_event_ids": [event["event_id"] for event in source_events],
            "preserved_recent_groups": self.preserve_recent_groups,
            "original_tokens": original_tokens,
            "final_tokens": final_tokens,
            "removed_tokens": original_tokens - final_tokens,
            "removed_groups": len(old_groups),
            "removed_messages": sum(len(group) for group in old_groups),
            "cache_hits": cache_hits,
            "provider_metadata": metadata,
        }
