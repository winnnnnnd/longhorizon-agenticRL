"""Sequential segment extraction and deterministic experience clustering.

The offline experience workflow intentionally keeps three representations
separate: actor-visible trajectory segments, model-formatted segment records,
and reusable Experience Cards.  This module owns the middle representation so
that clustering never depends on free-form whole-trajectory summaries.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
import json
import re

from shopping_grpo.experience.contracts import (
    DECISION_PHASES,
    EXPERIENCE_CARD_VERSION,
    EXPERIENCE_TYPES,
    RECALL_EVENTS,
    SCOPE_CONTRACT,
    TRIGGER_PREDICATES,
    canonical_sha256,
    finalize_experience_card,
    validate_experience_card,
)
from shopping_grpo.experience.segmentation import (
    SEGMENT_VERSION,
    build_trajectory_segments,
)


SEGMENT_EXTRACTION_INPUT_VERSION = "shopping-segment-extraction-input-v1"
SEGMENT_EXTRACTION_VERSION = "shopping-segment-extraction-v1"
SEGMENT_EXTRACTION_PROMPT_VERSION = "shopping-segment-format-v1"
SEGMENT_CLUSTER_VERSION = "shopping-segment-cluster-v1"
CLUSTER_DISTILLATION_VERSION = "shopping-segment-cluster-distillation-v1"
CLUSTER_DISTILLATION_PROMPT_VERSION = "shopping-segment-cluster-to-card-v1"

CONSTRAINT_TAGS = frozenset(
    {"category", "budget", "brand", "model", "core_function", "option"}
)
ACTION_PATTERNS = frozenset(
    {
        "task_parse",
        "initial_search",
        "search_reformulation",
        "paginate_results",
        "open_candidate",
        "compare_candidates",
        "inspect_evidence",
        "select_option",
        "verify_variant_price",
        "purchase",
        "abstain",
        "guard_recovery",
        "repeat_action",
        "mixed",
    }
)
FAILURE_TYPES = frozenset(
    {
        "none",
        "search_core_requirement_missed",
        "search_stagnation",
        "search_ineffective_reformulation",
        "candidate_selection_error",
        "candidate_comparison_insufficient",
        "candidate_overexploration",
        "critical_evidence_missing",
        "final_price_unverified",
        "option_selection_error",
        "premature_purchase",
        "premature_abstain",
        "guard_rejection",
        "repeat_loop",
        "max_steps_exhaustion",
        "context_loss",
        "other",
    }
)
LOCAL_OUTCOMES = frozenset(
    {
        "progress",
        "no_progress",
        "recovered",
        "terminal_success",
        "terminal_failure",
        "unknown",
    }
)
KNOWLEDGE_KINDS = frozenset(
    {"constraint", "candidate", "evidence", "strategy", "risk"}
)

_PRODUCT_ID = re.compile(r"(?<!\d)\d{8,12}(?!\d)")
_EXPLICIT_PRICE = re.compile(r"(?:[¥￥]\s*\d|\d+(?:\.\d+)?\s*元)")
_FORBIDDEN_TERMS = (
    "target_asin",
    "gold_purchase",
    "reward_detail",
    "audit_only_raw_observation",
)


SEGMENT_SYSTEM_PROMPT = f"""你是 Shopping Agent 的轨迹 Segment 格式化抽取器。
你每次只分析一个确定性切分的连续 Segment，但输入还会提供相邻事件和之前 Segment
已经确认的知识。请解释这个阶段的进入状态、决策、状态变化和局部结果，并将可传递
知识留给后续 Segment。不要对整条轨迹做泛泛总结。

只输出 JSON，schema_version 必须为 {SEGMENT_EXTRACTION_VERSION}。phase 和 segment_id
必须原样返回。failure_type、action_pattern、local_outcome、constraint_tags 和
state_predicates 只能使用输入枚举。所有 evidence_event_ids 必须来自输入给出的事件，
且至少引用当前 Segment 的一个事件。prior_knowledge_refs 只能引用输入提供的 knowledge_id。

输出内容只能基于 Actor-visible evidence。禁止写入商品 ID、具体价格、Gold 商品、Reward
字段、隐藏目标或 raw observation。没有失败证据时 failure_type 使用 none；不要因为最终
失败就臆测当前 Segment 一定有错。carried_knowledge 只保留后续决策真正需要的抽象知识。
不要输出 Markdown。"""


CLUSTER_SYSTEM_PROMPT = f"""你是 Shopping Agent 的 Experience Card 合成器。输入是由
相同六维 cluster key 聚合的 SegmentExtraction。请只总结被成员事件支持、可以跨任务
复用的程序性经验，不要复述某个商品或任务答案。

只输出 JSON，schema_version 必须为 {CLUSTER_DISTILLATION_VERSION}。experience 可以为
null；证据不足或只能得到任务特定事实时必须返回 null。否则 Experience Card 必须描述
何时适用、应该做什么、避免什么以及如何确认完成。supporting_segment_ids 和
contradicting_segment_ids 只能引用输入成员。禁止商品 ID、具体价格、Gold、Reward、隐藏
字段以及与当前工具或 Action Guard 冲突的建议。不要输出 Markdown。"""


def _mapping(value: object, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _exact_fields(value: Mapping, expected: set[str], path: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unexpected = sorted(set(value) - expected)
        raise ValueError(
            f"{path} fields differ from schema; missing={missing}, unexpected={unexpected}"
        )


def _text(value: object, path: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be non-empty text")
    return value.strip()


def _text_list(
    value: object,
    path: str,
    *,
    allowed: Iterable[str] | None = None,
    maximum: int = 64,
    allow_empty: bool = True,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    if not allow_empty and not value:
        raise ValueError(f"{path} cannot be empty")
    if len(value) > int(maximum):
        raise ValueError(f"{path} contains too many values")
    result = [_text(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise ValueError(f"{path} contains duplicate values")
    if allowed is not None and not set(result) <= set(allowed):
        unknown = sorted(set(result) - set(allowed))
        raise ValueError(f"{path} contains unsupported values: {unknown}")
    return result


def _safe_semantic_text(value: object, path: str) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if _PRODUCT_ID.search(payload):
        raise ValueError(f"{path} contains a product ID")
    if _EXPLICIT_PRICE.search(payload):
        raise ValueError(f"{path} contains a task-specific price")
    folded = payload.casefold()
    leaked = [term for term in _FORBIDDEN_TERMS if term in folded]
    if leaked:
        raise ValueError(f"{path} contains forbidden terms: {sorted(leaked)}")


def _event_ids(segment: Mapping, field: str) -> list[str]:
    return [
        str(event.get("event_id") or "")
        for event in segment.get(field) or []
        if isinstance(event, Mapping) and event.get("event_id")
    ]


def _terminal_outcome(segment: Mapping) -> str:
    outcome = segment.get("outcome")
    outcome = outcome if isinstance(outcome, Mapping) else {}
    return str(outcome.get("reward_type") or "unknown")


def _validate_segment_response(
    result: object,
    *,
    segment: Mapping,
    prior_knowledge: list[Mapping],
) -> dict:
    value = dict(_mapping(result, "segment_extraction"))
    expected_fields = {
        "schema_version",
        "segment_id",
        "phase",
        "summary",
        "state_transition",
        "constraint_tags",
        "state_predicates",
        "action_pattern",
        "failure_type",
        "local_outcome",
        "prior_knowledge_refs",
        "positive_pattern",
        "anti_pattern",
        "carried_knowledge",
        "evidence_event_ids",
    }
    _exact_fields(value, expected_fields, "segment_extraction")
    if value["schema_version"] != SEGMENT_EXTRACTION_VERSION:
        raise ValueError("unsupported segment extraction schema_version")
    if value["segment_id"] != segment["segment_id"]:
        raise ValueError("segment extraction changed segment_id")
    if value["phase"] != segment["phase"]:
        raise ValueError("segment extraction changed deterministic phase")
    value["summary"] = _text(value["summary"], "segment_extraction.summary")

    transition = dict(_mapping(value["state_transition"], "state_transition"))
    _exact_fields(transition, {"before", "decision", "after"}, "state_transition")
    for field in ("before", "decision", "after"):
        transition[field] = _text(transition[field], f"state_transition.{field}")
    value["state_transition"] = transition

    value["constraint_tags"] = _text_list(
        value["constraint_tags"],
        "segment_extraction.constraint_tags",
        allowed=CONSTRAINT_TAGS,
        maximum=len(CONSTRAINT_TAGS),
    )
    value["state_predicates"] = _text_list(
        value["state_predicates"],
        "segment_extraction.state_predicates",
        allowed=TRIGGER_PREDICATES,
        maximum=len(TRIGGER_PREDICATES),
    )
    for field, allowed in (
        ("action_pattern", ACTION_PATTERNS),
        ("failure_type", FAILURE_TYPES),
        ("local_outcome", LOCAL_OUTCOMES),
    ):
        value[field] = _text(value[field], f"segment_extraction.{field}")
        if value[field] not in allowed:
            raise ValueError(f"segment_extraction.{field} is unsupported")

    prior_ids = {str(item.get("knowledge_id") or "") for item in prior_knowledge}
    value["prior_knowledge_refs"] = _text_list(
        value["prior_knowledge_refs"],
        "segment_extraction.prior_knowledge_refs",
        allowed=prior_ids,
        maximum=len(prior_ids),
    )
    value["positive_pattern"] = _text(
        value["positive_pattern"],
        "segment_extraction.positive_pattern",
        allow_none=True,
    )
    value["anti_pattern"] = _text(
        value["anti_pattern"],
        "segment_extraction.anti_pattern",
        allow_none=True,
    )

    available_event_ids = set(
        _event_ids(segment, "context_before")
        + _event_ids(segment, "events")
        + _event_ids(segment, "context_after")
    )
    owned_event_ids = set(_event_ids(segment, "events"))
    value["evidence_event_ids"] = _text_list(
        value["evidence_event_ids"],
        "segment_extraction.evidence_event_ids",
        allowed=available_event_ids,
        maximum=len(available_event_ids),
        allow_empty=False,
    )
    if not set(value["evidence_event_ids"]) & owned_event_ids:
        raise ValueError("segment extraction must cite an event owned by the segment")

    knowledge_rows = value["carried_knowledge"]
    if not isinstance(knowledge_rows, list) or len(knowledge_rows) > 8:
        raise ValueError("carried_knowledge must be a list with at most eight items")
    normalized_knowledge = []
    seen_statements = set()
    for index, raw in enumerate(knowledge_rows, start=1):
        row = dict(_mapping(raw, f"carried_knowledge[{index - 1}]"))
        _exact_fields(
            row,
            {"kind", "statement", "evidence_event_ids"},
            f"carried_knowledge[{index - 1}]",
        )
        row["kind"] = _text(row["kind"], f"carried_knowledge[{index - 1}].kind")
        if row["kind"] not in KNOWLEDGE_KINDS:
            raise ValueError("carried knowledge kind is unsupported")
        row["statement"] = _text(
            row["statement"], f"carried_knowledge[{index - 1}].statement"
        )
        if row["statement"] in seen_statements:
            raise ValueError("carried knowledge contains duplicate statements")
        seen_statements.add(row["statement"])
        row["evidence_event_ids"] = _text_list(
            row["evidence_event_ids"],
            f"carried_knowledge[{index - 1}].evidence_event_ids",
            allowed=available_event_ids,
            maximum=len(available_event_ids),
            allow_empty=False,
        )
        normalized_knowledge.append(row)
    value["carried_knowledge"] = normalized_knowledge

    _safe_semantic_text(
        {
            "summary": value["summary"],
            "state_transition": value["state_transition"],
            "positive_pattern": value["positive_pattern"],
            "anti_pattern": value["anti_pattern"],
            "carried_knowledge": value["carried_knowledge"],
        },
        "segment extraction semantic content",
    )
    return value


def segment_cluster_key(extraction: Mapping) -> dict:
    """Return the six-dimensional, code-owned clustering key."""

    return {
        "phase": str(extraction["phase"]),
        "failure_type": str(extraction["failure_type"]),
        "constraint_tags": sorted(set(extraction.get("constraint_tags") or [])),
        "state_predicates": sorted(set(extraction.get("state_predicates") or [])),
        "action_pattern": str(extraction["action_pattern"]),
        "terminal_outcome": str(extraction["terminal_outcome"]),
    }


def _merge_prior_knowledge(
    previous: Iterable[Mapping],
    additions: Iterable[Mapping],
    maximum: int,
) -> list[dict]:
    """Keep the newest evidence-backed instance of each abstract statement."""

    merged = {}
    for raw in [*previous, *additions]:
        row = deepcopy(dict(raw))
        key = (str(row.get("kind") or ""), str(row.get("statement") or ""))
        if key in merged:
            del merged[key]
        merged[key] = row
    return list(merged.values())[-int(maximum) :]


class SequentialSegmentExtractor:
    """Extract segments in order and carry grounded knowledge forward."""

    def __init__(
        self,
        client,
        *,
        context_events: int = 2,
        maximum_prior_knowledge: int = 24,
    ):
        if int(context_events) < 0:
            raise ValueError("context_events cannot be negative")
        if int(maximum_prior_knowledge) < 1:
            raise ValueError("maximum_prior_knowledge must be positive")
        self.client = client
        self.context_events = int(context_events)
        self.maximum_prior_knowledge = int(maximum_prior_knowledge)

    def extract_trajectory(
        self,
        trajectory: Mapping,
        *,
        actor_role: str,
        source_kind: str,
    ) -> tuple[list[dict], list[dict]]:
        segments = build_trajectory_segments(
            trajectory,
            source_kind=source_kind,
            actor_role=actor_role,
            context_events=self.context_events,
        )
        prior_knowledge = []
        extractions = []
        audits = []
        for segment in segments:
            extraction, audit = self.extract_segment(
                segment,
                prior_knowledge=prior_knowledge,
            )
            extractions.append(extraction)
            audits.append(audit)
            prior_knowledge = _merge_prior_knowledge(
                prior_knowledge,
                extraction["carried_knowledge"],
                self.maximum_prior_knowledge,
            )
        return extractions, audits

    def extract_segment(
        self,
        segment: Mapping,
        *,
        prior_knowledge: Iterable[Mapping] = (),
    ) -> tuple[dict, dict]:
        if segment.get("schema_version") != SEGMENT_VERSION:
            raise ValueError("unsupported trajectory segment schema")
        prior_rows = [deepcopy(dict(item)) for item in prior_knowledge]
        payload = {
            "schema_version": SEGMENT_EXTRACTION_INPUT_VERSION,
            "prompt_version": SEGMENT_EXTRACTION_PROMPT_VERSION,
            "allowed_values": {
                "constraint_tags": sorted(CONSTRAINT_TAGS),
                "state_predicates": sorted(TRIGGER_PREDICATES),
                "action_patterns": sorted(ACTION_PATTERNS),
                "failure_types": sorted(FAILURE_TYPES),
                "local_outcomes": sorted(LOCAL_OUTCOMES),
                "knowledge_kinds": sorted(KNOWLEDGE_KINDS),
            },
            "public_query": segment["public_query"],
            "segment": {
                key: deepcopy(segment[key])
                for key in (
                    "segment_id",
                    "segment_index",
                    "trajectory_id",
                    "task_id",
                    "actor_role",
                    "source_kind",
                    "phase",
                    "event_range",
                    "context_before",
                    "events",
                    "context_after",
                    "observed_state_predicates",
                )
            },
            "prior_segment_knowledge": prior_rows,
            "terminal_outcome": deepcopy(segment.get("outcome") or {}),
            "required_output": {
                "schema_version": SEGMENT_EXTRACTION_VERSION,
                "segment_id": segment["segment_id"],
                "phase": segment["phase"],
                "summary": "非空的阶段摘要",
                "state_transition": {
                    "before": "进入阶段时已知状态",
                    "decision": "本阶段采取的决策",
                    "after": "本阶段结束后的状态变化",
                },
                "constraint_tags": [],
                "state_predicates": [],
                "action_pattern": "allowed action_pattern",
                "failure_type": "allowed failure_type",
                "local_outcome": "allowed local_outcome",
                "prior_knowledge_refs": [],
                "positive_pattern": None,
                "anti_pattern": None,
                "carried_knowledge": [
                    {
                        "kind": "allowed knowledge_kind",
                        "statement": "供后续 Segment 使用的抽象知识",
                        "evidence_event_ids": ["e0001"],
                    }
                ],
                "evidence_event_ids": ["e0001"],
            },
        }
        response = self.client.complete_json(
            [
                {"role": "system", "content": SEGMENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                },
            ]
        )
        result = _validate_segment_response(
            response.get("result"),
            segment=segment,
            prior_knowledge=prior_rows,
        )
        result["state_predicates"] = sorted(
            set(result["state_predicates"])
            | set(segment.get("observed_state_predicates") or [])
        )
        knowledge = []
        for index, row in enumerate(result.pop("carried_knowledge"), start=1):
            knowledge.append(
                {
                    "knowledge_id": f"{segment['segment_id']}:k{index:03d}",
                    **row,
                }
            )
        extraction = {
            "schema_version": SEGMENT_EXTRACTION_VERSION,
            "segment_id": segment["segment_id"],
            "segment_index": int(segment["segment_index"]),
            "trajectory_id": segment["trajectory_id"],
            "task_id": int(segment["task_id"]),
            "actor_role": segment["actor_role"],
            "source_kind": segment["source_kind"],
            "public_query": segment["public_query"],
            "phase": segment["phase"],
            "event_range": deepcopy(segment["event_range"]),
            **{
                key: deepcopy(result[key])
                for key in (
                    "summary",
                    "state_transition",
                    "constraint_tags",
                    "state_predicates",
                    "action_pattern",
                    "failure_type",
                    "local_outcome",
                    "prior_knowledge_refs",
                    "positive_pattern",
                    "anti_pattern",
                    "evidence_event_ids",
                )
            },
            "observed_state_predicates": deepcopy(
                segment.get("observed_state_predicates") or []
            ),
            "carried_knowledge": knowledge,
            "terminal_outcome": _terminal_outcome(segment),
        }
        extraction["cluster_key"] = segment_cluster_key(extraction)
        extraction["cluster_key_hash"] = canonical_sha256(extraction["cluster_key"])
        audit = {
            "segment_id": segment["segment_id"],
            "input_hash": canonical_sha256(payload),
            "cluster_key_hash": extraction["cluster_key_hash"],
            "provider_metadata": deepcopy(response.get("metadata") or {}),
        }
        return extraction, audit


def cluster_segment_extractions(extractions: Iterable[Mapping]) -> list[dict]:
    """Merge records with an identical six-dimensional cluster key."""

    groups = {}
    seen_segments = set()
    for raw in extractions:
        row = deepcopy(dict(raw))
        if row.get("schema_version") != SEGMENT_EXTRACTION_VERSION:
            raise ValueError("cluster input has unsupported segment extraction schema")
        segment_id = str(row.get("segment_id") or "")
        if not segment_id or segment_id in seen_segments:
            raise ValueError("cluster input has missing or duplicate segment_id")
        seen_segments.add(segment_id)
        expected_key = segment_cluster_key(row)
        expected_hash = canonical_sha256(expected_key)
        if row.get("cluster_key") != expected_key or row.get("cluster_key_hash") != expected_hash:
            raise ValueError("segment extraction cluster key is inconsistent")
        groups.setdefault(expected_hash, []).append(row)

    clusters = []
    for key_hash in sorted(groups):
        members = sorted(
            groups[key_hash],
            key=lambda row: (
                int(row["task_id"]),
                str(row["trajectory_id"]),
                int(row["segment_index"]),
            ),
        )
        clusters.append(
            {
                "schema_version": SEGMENT_CLUSTER_VERSION,
                "cluster_id": "segment-cluster-" + key_hash[:16],
                "cluster_key_hash": key_hash,
                "cluster_key": deepcopy(members[0]["cluster_key"]),
                "member_count": len(members),
                "trajectory_ids": sorted(
                    {str(member["trajectory_id"]) for member in members}
                ),
                "source_counts": dict(
                    sorted(Counter(member["source_kind"] for member in members).items())
                ),
                "members": members,
            }
        )
    return clusters


def _cluster_source_kind(members: list[Mapping]) -> str:
    roles = {str(member["actor_role"]) for member in members}
    sources = {str(member["source_kind"]) for member in members}
    if roles == {"teacher", "student"}:
        return "teacher_agent_pair"
    if sources == {"curated_teacher_gold"}:
        return "curated_teacher_gold"
    if sources == {"teacher_raw_rollout"}:
        return "teacher_raw_rollout"
    return "agent_rollout"


def _representative_members(members: list[dict], maximum: int) -> list[dict]:
    """Select a deterministic, failure/outcome-balanced subset from a large cluster."""

    if maximum < 1:
        raise ValueError("maximum_cluster_members must be positive")
    ordered = sorted(
        members,
        key=lambda member: (
            str(member["actor_role"]),
            str(member["local_outcome"]),
            str(member["failure_type"]),
            int(member["task_id"]),
            str(member["trajectory_id"]),
            int(member["segment_index"]),
        ),
    )
    if len(ordered) <= maximum:
        return ordered
    buckets = {}
    for member in ordered:
        bucket = (
            str(member["actor_role"]),
            str(member["local_outcome"]),
            str(member["failure_type"]),
        )
        buckets.setdefault(bucket, []).append(member)
    selected = []
    while len(selected) < maximum:
        progressed = False
        for bucket in sorted(buckets):
            if buckets[bucket]:
                selected.append(buckets[bucket].pop(0))
                progressed = True
                if len(selected) == maximum:
                    break
        if not progressed:
            break
    return selected


class SegmentClusterDistiller:
    """Turn one homogeneous segment cluster into at most one candidate card."""

    def __init__(
        self,
        client,
        *,
        created_at: str | None = None,
        maximum_cluster_members: int = 16,
    ):
        self.client = client
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.maximum_cluster_members = int(maximum_cluster_members)
        if self.maximum_cluster_members < 1:
            raise ValueError("maximum_cluster_members must be positive")

    def distill_cluster(self, cluster: Mapping) -> tuple[dict | None, dict]:
        if cluster.get("schema_version") != SEGMENT_CLUSTER_VERSION:
            raise ValueError("unsupported segment cluster schema")
        all_members = [deepcopy(dict(item)) for item in cluster.get("members") or []]
        if not all_members:
            raise ValueError("segment cluster cannot be empty")
        members = _representative_members(
            all_members,
            self.maximum_cluster_members,
        )
        input_cluster = {
            key: deepcopy(value)
            for key, value in cluster.items()
            if key != "members"
        }
        input_cluster.update(
            {
                "total_member_count": len(all_members),
                "selected_member_count": len(members),
                "members": members,
            }
        )
        payload = {
            "schema_version": SEGMENT_CLUSTER_VERSION,
            "prompt_version": CLUSTER_DISTILLATION_PROMPT_VERSION,
            "allowed_experience_types": sorted(EXPERIENCE_TYPES),
            "allowed_recall_events": sorted(RECALL_EVENTS),
            "allowed_predicates": sorted(TRIGGER_PREDICATES),
            "cluster": input_cluster,
            "required_output": {
                "schema_version": CLUSTER_DISTILLATION_VERSION,
                "analysis": {
                    "shared_pattern": "成员共同支持的模式",
                    "applicability": "适用边界",
                },
                "experience": {
                    "type": "allowed experience type",
                    "phase": cluster["cluster_key"]["phase"],
                    "trigger": {
                        "events": ["allowed recall event"],
                        "page_types": [],
                        "required_constraint_tags": [],
                        "predicates": [],
                    },
                    "guidance": ["触发后应采取的动作"],
                    "anti_patterns": ["应该避免的动作"],
                    "verification_checks": ["完成验证条件"],
                    "categories": [],
                    "supporting_segment_ids": [members[0]["segment_id"]],
                    "contradicting_segment_ids": [],
                },
                "experience_may_be_null": True,
            },
        }
        response = self.client.complete_json(
            [
                {"role": "system", "content": CLUSTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                },
            ]
        )
        result = dict(_mapping(response.get("result"), "cluster_distillation"))
        _exact_fields(
            result,
            {"schema_version", "analysis", "experience"},
            "cluster_distillation",
        )
        if result["schema_version"] != CLUSTER_DISTILLATION_VERSION:
            raise ValueError("unsupported cluster distillation schema_version")
        analysis = dict(_mapping(result["analysis"], "cluster_distillation.analysis"))
        _exact_fields(
            analysis,
            {"shared_pattern", "applicability"},
            "cluster_distillation.analysis",
        )
        for field in ("shared_pattern", "applicability"):
            analysis[field] = _text(
                analysis[field],
                f"cluster_distillation.analysis.{field}",
            )
        _safe_semantic_text(analysis, "cluster distillation analysis")
        raw_experience = result["experience"]
        if raw_experience is None:
            return None, {
                "cluster_id": cluster["cluster_id"],
                "input_hash": canonical_sha256(payload),
                "total_member_count": len(all_members),
                "selected_segment_ids": [member["segment_id"] for member in members],
                "analysis": analysis,
                "experience_id": None,
                "content_hash": None,
                "lineage": [],
                "provider_metadata": deepcopy(response.get("metadata") or {}),
            }

        item = dict(_mapping(raw_experience, "cluster_distillation.experience"))
        expected_fields = {
            "type",
            "phase",
            "trigger",
            "guidance",
            "anti_patterns",
            "verification_checks",
            "categories",
            "supporting_segment_ids",
            "contradicting_segment_ids",
        }
        _exact_fields(item, expected_fields, "cluster_distillation.experience")
        if item["type"] not in EXPERIENCE_TYPES:
            raise ValueError("cluster experience type is unsupported")
        if item["phase"] != cluster["cluster_key"]["phase"]:
            raise ValueError("cluster experience changed the deterministic phase")
        if item["phase"] not in DECISION_PHASES:
            raise ValueError("cluster experience phase is unsupported")
        trigger = dict(_mapping(item["trigger"], "cluster_distillation.trigger"))
        _exact_fields(
            trigger,
            {"events", "page_types", "required_constraint_tags", "predicates"},
            "cluster_distillation.trigger",
        )
        trigger["required_constraint_tags"] = _text_list(
            trigger["required_constraint_tags"],
            "cluster_distillation.trigger.required_constraint_tags",
            allowed=CONSTRAINT_TAGS,
            maximum=len(CONSTRAINT_TAGS),
        )
        trigger["predicates"] = _text_list(
            trigger["predicates"],
            "cluster_distillation.trigger.predicates",
            allowed=TRIGGER_PREDICATES,
            maximum=len(TRIGGER_PREDICATES),
        )
        if not set(trigger["required_constraint_tags"]) <= set(
            cluster["cluster_key"]["constraint_tags"]
        ):
            raise ValueError("cluster experience invented a constraint trigger")
        if not set(trigger["predicates"]) <= set(
            cluster["cluster_key"]["state_predicates"]
        ):
            raise ValueError("cluster experience invented a state predicate trigger")
        item["trigger"] = trigger
        valid_segment_ids = {str(member["segment_id"]) for member in members}
        supporting = _text_list(
            item["supporting_segment_ids"],
            "experience.supporting_segment_ids",
            allowed=valid_segment_ids,
            maximum=len(valid_segment_ids),
            allow_empty=False,
        )
        contradicting = _text_list(
            item["contradicting_segment_ids"],
            "experience.contradicting_segment_ids",
            allowed=valid_segment_ids,
            maximum=len(valid_segment_ids),
        )
        if set(supporting) & set(contradicting):
            raise ValueError("supporting and contradicting segments overlap")
        member_by_id = {str(member["segment_id"]): member for member in members}
        supporting_trajectories = sorted(
            {str(member_by_id[segment_id]["trajectory_id"]) for segment_id in supporting}
        )
        contradicting_trajectories = sorted(
            {
                str(member_by_id[segment_id]["trajectory_id"])
                for segment_id in contradicting
            }
            - set(supporting_trajectories)
        )
        cluster_key_hash = str(cluster["cluster_key_hash"])
        experience_id = f"exp-{item['type'].replace('_', '-')}-{cluster_key_hash[:12]}"
        semantic_seed = {
            "type": item["type"],
            "phase": item["phase"],
            "trigger": deepcopy(item["trigger"]),
            "guidance": deepcopy(item["guidance"]),
            "anti_patterns": deepcopy(item["anti_patterns"]),
            "verification_checks": deepcopy(item["verification_checks"]),
            "scope": {**SCOPE_CONTRACT, "categories": deepcopy(item["categories"])},
        }
        _safe_semantic_text(semantic_seed, "cluster experience semantic content")
        card = finalize_experience_card(
            {
                "schema_version": EXPERIENCE_CARD_VERSION,
                "experience_id": experience_id,
                "revision": 1,
                "status": "candidate",
                **semantic_seed,
                "evidence": {
                    "source_kind": _cluster_source_kind(members),
                    "supporting_trajectory_ids": supporting_trajectories,
                    "contradicting_trajectory_ids": contradicting_trajectories,
                    "support_count": len(supporting_trajectories),
                },
                "provenance": {
                    "extractor": str(self.client.model),
                    "extractor_revision": str(self.client.model_revision),
                    "prompt_version": CLUSTER_DISTILLATION_PROMPT_VERSION,
                    "created_at": self.created_at,
                },
                "supersedes": None,
                "content_hash": "pending",
            }
        )
        lineage = [
            {
                "segment_id": segment_id,
                "trajectory_id": member_by_id[segment_id]["trajectory_id"],
                "event_ids": deepcopy(member_by_id[segment_id]["evidence_event_ids"]),
                "role": (
                    "supporting" if segment_id in supporting else "contradicting"
                ),
            }
            for segment_id in supporting + contradicting
        ]
        audit = {
            "cluster_id": cluster["cluster_id"],
            "input_hash": canonical_sha256(payload),
            "total_member_count": len(all_members),
            "selected_segment_ids": [member["segment_id"] for member in members],
            "analysis": analysis,
            "experience_id": card["experience_id"],
            "content_hash": card["content_hash"],
            "lineage": lineage,
            "provider_metadata": deepcopy(response.get("metadata") or {}),
        }
        return card, audit


def apply_latest_card_revisions(
    cards: Iterable[Mapping],
    previous_cards: Iterable[Mapping] = (),
) -> list[dict]:
    """Make the newest card revision supersede the prior card with the same ID."""

    previous = [validate_experience_card(card) for card in previous_cards]
    latest_previous = {}
    for card in previous:
        current = latest_previous.get(card["experience_id"])
        if current is None or card["revision"] > current["revision"]:
            latest_previous[card["experience_id"]] = card

    result = []
    seen_ids = set()
    for raw in cards:
        card = validate_experience_card(raw)
        experience_id = card["experience_id"]
        if experience_id in seen_ids:
            raise ValueError("new cards contain duplicate experience IDs")
        seen_ids.add(experience_id)
        prior = latest_previous.get(experience_id)
        updated = deepcopy(card)
        if prior is not None:
            updated["revision"] = int(prior["revision"]) + 1
            updated["supersedes"] = f"{experience_id}@{prior['revision']}"
            supporting = sorted(
                set(prior["evidence"]["supporting_trajectory_ids"])
                | set(updated["evidence"]["supporting_trajectory_ids"])
            )
            contradicting = sorted(
                (
                    set(prior["evidence"]["contradicting_trajectory_ids"])
                    | set(updated["evidence"]["contradicting_trajectory_ids"])
                )
                - set(supporting)
            )
            updated["evidence"]["supporting_trajectory_ids"] = supporting
            updated["evidence"]["contradicting_trajectory_ids"] = contradicting
            updated["evidence"]["support_count"] = len(supporting)
        result.append(finalize_experience_card(updated))
    return sorted(result, key=lambda card: (card["experience_id"], card["revision"]))
