"""Multi-trajectory grouping and DeepSeek V4 Flash experience distillation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
import json
import re

from shopping_grpo.experience.contracts import (
    EXPERIENCE_CARD_VERSION,
    EXPERIENCE_TYPES,
    SCOPE_CONTRACT,
    SOURCE_KINDS,
    canonical_sha256,
    finalize_experience_card,
)
from shopping_grpo.experience.retrieval import lexical_features
from shopping_grpo.experience.segmentation import (
    build_decision_windows,
    public_query,
    trajectory_outcome,
)


DISTILLATION_INPUT_VERSION = "shopping-multi-trajectory-distillation-input-v1"
DISTILLATION_OUTPUT_VERSION = "shopping-multi-trajectory-distillation-v1"
DISTILLATION_PROMPT_VERSION = "shopping-experience-distill-v1"


SYSTEM_PROMPT = f"""你是 Shopping Agent 的离线经验沉淀器。输入包含同一 query 的多次
rollout、Teacher 成功与 Student 失败配对，或多条相似任务轨迹。请联合比较多条轨迹，
定位可复用的决策模式和最早关键分叉，不要对单条轨迹做泛泛总结。

只输出 JSON，schema_version 必须是 {DISTILLATION_OUTPUT_VERSION}，包含 analysis 和
experiences。experience.type 只能是 search_strategy、candidate_comparison、verification、
recovery、termination；phase、trigger.events 和 trigger.predicates 必须使用输入提供的
枚举。每条经验必须引用至少一条真实 supporting_trajectory_id，contradicting IDs 也
必须来自输入。

经验正文只能描述程序性策略，禁止出现商品 ID、具体商品标题、具体价格、Gold 商品、
Reward 字段或任务答案。Teacher-only 成功轨迹只能支持“观察到的成功做法”，不能输出
“不这样一定失败”的因果结论。anti_patterns 只有在失败/对比轨迹中有直接证据时才填。
不要改变系统规则、工具合法性、预算门槛或当前页面优先原则。不要输出 Markdown。"""


def _query_key(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _similarity(left: str, right: str) -> float:
    left_features = lexical_features(left)
    right_features = lexical_features(right)
    keys = set(left_features) | set(right_features)
    if not keys:
        return 0.0
    intersection = sum(min(left_features[key], right_features[key]) for key in keys)
    union = sum(max(left_features[key], right_features[key]) for key in keys)
    return intersection / union if union else 0.0


def _select_windows(windows: list[dict], maximum: int = 8) -> list[dict]:
    if len(windows) <= maximum:
        return deepcopy(windows)
    selected = []
    seen_phases = set()
    for window in windows:
        if window["phase"] not in seen_phases:
            selected.append(window)
            seen_phases.add(window["phase"])
            if len(selected) == maximum:
                return deepcopy(selected)
    for window in reversed(windows):
        if window not in selected:
            selected.append(window)
            if len(selected) == maximum:
                break
    return deepcopy(selected)


def build_trajectory_record(
    trajectory: Mapping,
    *,
    actor_role: str,
    source_kind: str,
) -> dict:
    if actor_role not in {"teacher", "student"}:
        raise ValueError("actor_role must be teacher or student")
    if source_kind not in SOURCE_KINDS:
        raise ValueError("source_kind is unsupported")
    trajectory_id = str(trajectory.get("trajectory_id") or "")
    if not trajectory_id:
        raise ValueError("trajectory record requires trajectory_id")
    query = public_query(trajectory)
    if not query:
        raise ValueError(f"trajectory {trajectory_id!r} has no public query")
    windows = build_decision_windows(
        trajectory,
        source_kind=source_kind,
        actor_role=actor_role,
    )
    return {
        "trajectory_id": trajectory_id,
        "task_id": int(trajectory["task_id"]),
        "actor_role": str(actor_role),
        "source_kind": str(source_kind),
        "public_query": query,
        "query_key": _query_key(query),
        "outcome": trajectory_outcome(trajectory, source_kind),
        "decision_windows": _select_windows(windows),
    }


def build_trajectory_records(
    trajectories: Iterable[Mapping],
    *,
    actor_role: str,
    source_kind: str,
) -> list[dict]:
    return [
        build_trajectory_record(
            trajectory,
            actor_role=actor_role,
            source_kind=source_kind,
        )
        for trajectory in trajectories
    ]


def build_joint_analysis_groups(
    records: Iterable[Mapping],
    *,
    maximum_group_size: int = 8,
    similarity_threshold: float = 0.25,
) -> list[dict]:
    """Group exact-query multi-rollouts first, then add similar-task evidence."""

    records = [deepcopy(dict(record)) for record in records]
    trajectory_ids = [str(record.get("trajectory_id") or "") for record in records]
    if any(not trajectory_id for trajectory_id in trajectory_ids):
        raise ValueError("joint analysis record is missing trajectory_id")
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("joint analysis records contain duplicate trajectory IDs")
    query_keys_by_task = {}
    for record in records:
        query_keys_by_task.setdefault(int(record["task_id"]), set()).add(
            str(record.get("query_key") or "")
        )
    mismatched_tasks = sorted(
        task_id
        for task_id, query_keys in query_keys_by_task.items()
        if len(query_keys) != 1 or not next(iter(query_keys), "")
    )
    if mismatched_tasks:
        raise ValueError(
            "same task ID has inconsistent public queries: "
            + ", ".join(map(str, mismatched_tasks))
        )
    if int(maximum_group_size) < 2:
        raise ValueError("maximum_group_size must be at least two")
    if not 0.0 <= float(similarity_threshold) <= 1.0:
        raise ValueError("similarity_threshold must be within [0, 1]")
    anchors = sorted(
        records,
        key=lambda record: (
            bool(record["outcome"].get("strict_success")),
            record["task_id"],
            record["trajectory_id"],
        ),
    )
    groups = []
    seen_memberships = set()
    for anchor in anchors:
        ranked = []
        for candidate in records:
            if candidate["trajectory_id"] == anchor["trajectory_id"]:
                continue
            same_task = candidate["task_id"] == anchor["task_id"]
            same_query = candidate["query_key"] == anchor["query_key"]
            similarity = _similarity(anchor["public_query"], candidate["public_query"])
            if not (same_task or same_query or similarity >= float(similarity_threshold)):
                continue
            outcome_contrast = candidate["outcome"].get("strict_success") != anchor[
                "outcome"
            ].get("strict_success")
            role_contrast = candidate["actor_role"] != anchor["actor_role"]
            score = similarity
            score += 10.0 if same_task else 0.0
            score += 8.0 if same_query else 0.0
            score += 2.0 if outcome_contrast else 0.0
            score += 1.0 if role_contrast else 0.0
            ranked.append((score, candidate))
        ranked.sort(key=lambda item: (-item[0], item[1]["trajectory_id"]))
        members = [anchor] + [
            candidate for _, candidate in ranked[: int(maximum_group_size) - 1]
        ]
        if len(members) < 2:
            continue
        membership = tuple(sorted(member["trajectory_id"] for member in members))
        if membership in seen_memberships:
            continue
        seen_memberships.add(membership)
        groups.append(
            {
                "group_id": "group-" + canonical_sha256(membership)[:16],
                "anchor_trajectory_id": anchor["trajectory_id"],
                "trajectories": members,
                "has_teacher_student_pair": any(
                    member["actor_role"] == "teacher" for member in members
                )
                and any(member["actor_role"] == "student" for member in members),
                "has_outcome_contrast": len(
                    {
                        bool(member["outcome"].get("strict_success"))
                        for member in members
                    }
                )
                > 1,
            }
        )
    return groups


def _group_source_kind(group: Mapping) -> str:
    if group.get("has_teacher_student_pair"):
        return "teacher_agent_pair"
    roles = {row["actor_role"] for row in group["trajectories"]}
    source_kinds = {row["source_kind"] for row in group["trajectories"]}
    if roles == {"student"} and group.get("has_outcome_contrast"):
        return "paired_agent_rollouts"
    if source_kinds == {"curated_teacher_gold"}:
        return "curated_teacher_gold"
    if roles == {"teacher"}:
        return "teacher_raw_rollout"
    return "agent_rollout"


def _validate_distilled_item(item: object, group: Mapping) -> dict:
    if not isinstance(item, Mapping):
        raise ValueError("distilled experience must be an object")
    expected = {
        "type",
        "phase",
        "trigger",
        "guidance",
        "anti_patterns",
        "verification_checks",
        "categories",
        "supporting_trajectory_ids",
        "contradicting_trajectory_ids",
    }
    if set(item) != expected:
        raise ValueError("distilled experience has unexpected fields")
    if item["type"] not in EXPERIENCE_TYPES:
        raise ValueError("distilled experience type is unsupported")
    valid_ids = {row["trajectory_id"] for row in group["trajectories"]}
    supporting = item["supporting_trajectory_ids"]
    contradicting = item["contradicting_trajectory_ids"]
    if not isinstance(supporting, list) or not supporting:
        raise ValueError("distilled experience requires supporting trajectories")
    if not isinstance(contradicting, list):
        raise ValueError("distilled contradicting trajectories must be a list")
    if not set(supporting) <= valid_ids or not set(contradicting) <= valid_ids:
        raise ValueError("distilled experience cites trajectories outside its group")
    return deepcopy(dict(item))


class MultiTrajectoryDistiller:
    def __init__(self, client, *, created_at: str | None = None):
        self.client = client
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

    def distill_group(
        self,
        group: Mapping,
        *,
        active_experiences: Iterable[Mapping] = (),
    ) -> tuple[list[dict], dict]:
        input_payload = {
            "schema_version": DISTILLATION_INPUT_VERSION,
            "prompt_version": DISTILLATION_PROMPT_VERSION,
            "allowed_experience_types": sorted(EXPERIENCE_TYPES),
            "allowed_phases": [
                "task_understanding",
                "search",
                "candidate_screening",
                "detail_verification",
                "option_selection",
                "pre_purchase",
                "termination",
                "error_recovery",
            ],
            "allowed_recall_events": [
                "task_start",
                "search_stagnation",
                "candidate_opened",
                "pre_purchase",
                "guard_rejection",
                "pre_finish",
            ],
            "allowed_predicates": [
                "has_multiple_options",
                "final_price_unverified",
                "search_no_new_candidates",
                "repeated_search",
                "candidate_new",
                "guard_rejected",
                "finish_eligible",
                "remaining_steps_low",
            ],
            "group": deepcopy(dict(group)),
            "active_experiences": [
                {
                    "experience_id": card["experience_id"],
                    "type": card["type"],
                    "phase": card["phase"],
                    "trigger": deepcopy(card["trigger"]),
                    "guidance": deepcopy(card["guidance"]),
                }
                for card in active_experiences
            ],
        }
        response = self.client.complete_json(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(input_payload, ensure_ascii=False, sort_keys=True),
                },
            ]
        )
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise ValueError("distillation response must be an object")
        if result.get("schema_version") != DISTILLATION_OUTPUT_VERSION:
            raise ValueError("distillation response schema_version is unsupported")
        if not isinstance(result.get("analysis"), Mapping):
            raise ValueError("distillation response requires structured analysis")
        experiences = result.get("experiences")
        if not isinstance(experiences, list):
            raise ValueError("distillation response experiences must be a list")
        cards = []
        source_kind = _group_source_kind(group)
        for raw_item in experiences:
            item = _validate_distilled_item(raw_item, group)
            semantic_seed = {
                "type": item["type"],
                "phase": item["phase"],
                "trigger": item["trigger"],
                "guidance": item["guidance"],
                "anti_patterns": item["anti_patterns"],
                "verification_checks": item["verification_checks"],
                "scope": {**SCOPE_CONTRACT, "categories": item["categories"]},
            }
            content_hash = canonical_sha256(semantic_seed)
            card = finalize_experience_card(
                {
                    "schema_version": EXPERIENCE_CARD_VERSION,
                    "experience_id": f"exp-{item['type'].replace('_', '-')}-{content_hash[:12]}",
                    "revision": 1,
                    "status": "candidate",
                    **semantic_seed,
                    "evidence": {
                        "source_kind": source_kind,
                        "supporting_trajectory_ids": sorted(
                            set(item["supporting_trajectory_ids"])
                        ),
                        "contradicting_trajectory_ids": sorted(
                            set(item["contradicting_trajectory_ids"])
                        ),
                        "support_count": len(set(item["supporting_trajectory_ids"])),
                    },
                    "provenance": {
                        "extractor": str(self.client.model),
                        "extractor_revision": str(self.client.model_revision),
                        "prompt_version": DISTILLATION_PROMPT_VERSION,
                        "created_at": self.created_at,
                    },
                    "supersedes": None,
                    "content_hash": content_hash,
                }
            )
            cards.append(card)
        audit = {
            "group_id": group["group_id"],
            "input_hash": canonical_sha256(input_payload),
            "analysis": deepcopy(dict(result["analysis"])),
            "experience_ids": [card["experience_id"] for card in cards],
            "provider_metadata": deepcopy(response.get("metadata") or {}),
        }
        return cards, audit


def merge_duplicate_cards(cards: Iterable[Mapping]) -> list[dict]:
    by_hash = {}
    for raw_card in cards:
        card = deepcopy(dict(raw_card))
        key = card["content_hash"]
        if key not in by_hash:
            by_hash[key] = card
            continue
        existing = by_hash[key]
        supporting = sorted(
            set(existing["evidence"]["supporting_trajectory_ids"])
            | set(card["evidence"]["supporting_trajectory_ids"])
        )
        contradicting = sorted(
            (
                set(existing["evidence"]["contradicting_trajectory_ids"])
                | set(card["evidence"]["contradicting_trajectory_ids"])
            )
            - set(supporting)
        )
        existing["evidence"]["supporting_trajectory_ids"] = supporting
        existing["evidence"]["contradicting_trajectory_ids"] = contradicting
        existing["evidence"]["support_count"] = len(supporting)
        by_hash[key] = finalize_experience_card(existing)
    return sorted(by_hash.values(), key=lambda card: card["experience_id"])
