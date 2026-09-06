"""Strict public contracts for external experience cards and runtime bundles."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re


EXPERIENCE_CARD_VERSION = "shopping-experience-card-v1"
EXPERIENCE_STORE_MANIFEST_VERSION = "shopping-experience-store-manifest-v1"
EXPERIENCE_RETRIEVER_VERSION = "shopping-experience-retriever-v1"
EXPERIENCE_INJECTION_VERSION = "experience-guidance-v1"

EXPERIENCE_TYPES = frozenset(
    {
        "search_strategy",
        "candidate_comparison",
        "verification",
        "recovery",
        "termination",
    }
)
DECISION_PHASES = frozenset(
    {
        "task_understanding",
        "search",
        "candidate_screening",
        "detail_verification",
        "option_selection",
        "pre_purchase",
        "termination",
        "error_recovery",
    }
)
RECALL_EVENTS = frozenset(
    {
        "task_start",
        "search_stagnation",
        "candidate_opened",
        "pre_purchase",
        "guard_rejection",
        "pre_finish",
    }
)
RECALL_EVENT_PHASES = {
    "task_start": frozenset({"task_understanding", "search"}),
    "search_stagnation": frozenset({"search"}),
    "candidate_opened": frozenset({"candidate_screening", "detail_verification"}),
    "pre_purchase": frozenset(
        {"detail_verification", "option_selection", "pre_purchase"}
    ),
    "guard_rejection": frozenset({"error_recovery"}),
    "pre_finish": frozenset({"termination"}),
}
PAGE_TYPES = frozenset(
    {
        "search_home",
        "search_results",
        "product_detail",
        "information_subpage",
        "terminal",
        "unknown",
    }
)
CARD_STATUSES = frozenset(
    {"candidate", "validated", "active", "deprecated", "rejected"}
)
SOURCE_KINDS = frozenset(
    {
        "curated_teacher_gold",
        "teacher_raw_rollout",
        "agent_rollout",
        "paired_agent_rollouts",
        "teacher_agent_pair",
    }
)
TRIGGER_PREDICATES = frozenset(
    {
        "has_multiple_options",
        "final_price_unverified",
        "search_no_new_candidates",
        "repeated_search",
        "candidate_new",
        "guard_rejected",
        "finish_eligible",
        "remaining_steps_low",
    }
)
SCOPE_CONTRACT = {
    "environment_version": "shopsimulator-environment-v2.1",
    "reward_version": "shopsimulator-reward-v3",
    "observation_version": "shopping-observation-v2",
    "tool_version": "shopping-tools-v2",
}
SEMANTIC_CONTENT_FIELDS = (
    "type",
    "phase",
    "trigger",
    "guidance",
    "anti_patterns",
    "verification_checks",
    "scope",
)
_EXPERIENCE_ID = re.compile(r"^exp-[a-z0-9][a-z0-9-]{5,95}$")
_PRODUCT_ID = re.compile(r"(?<!\d)\d{8,12}(?!\d)")
_EXPLICIT_PRICE = re.compile(r"(?:[¥￥]\s*\d|\d+(?:\.\d+)?\s*元)")
_FORBIDDEN_SEMANTIC_TERMS = (
    "target_asin",
    "gold_purchase",
    "reward_detail",
    "audit_only_raw_observation",
)


class ExperienceContractError(ValueError):
    """Raised when experience data is malformed, unsafe or non-reproducible."""


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_mapping(value: object, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ExperienceContractError(f"{path} must be an object")
    return value


def _require_nonempty_text(value: object, path: str, *, maximum: int = 1000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperienceContractError(f"{path} must be non-empty text")
    text = value.strip()
    if len(text) > maximum:
        raise ExperienceContractError(f"{path} exceeds {maximum} characters")
    return text


def _text_list(
    value: object,
    path: str,
    *,
    maximum_items: int,
    allow_empty: bool = True,
    item_maximum: int = 500,
) -> list[str]:
    if not isinstance(value, list):
        raise ExperienceContractError(f"{path} must be a list")
    if not allow_empty and not value:
        raise ExperienceContractError(f"{path} cannot be empty")
    if len(value) > maximum_items:
        raise ExperienceContractError(
            f"{path} cannot contain more than {maximum_items} items"
        )
    result = [
        _require_nonempty_text(item, f"{path}[{index}]", maximum=item_maximum)
        for index, item in enumerate(value)
    ]
    if len(result) != len(set(result)):
        raise ExperienceContractError(f"{path} contains duplicate values")
    return result


def experience_semantic_content(card: Mapping) -> dict:
    return {field: deepcopy(card.get(field)) for field in SEMANTIC_CONTENT_FIELDS}


def experience_content_hash(card: Mapping) -> str:
    return canonical_sha256(experience_semantic_content(card))


def finalize_experience_card(card: Mapping) -> dict:
    result = deepcopy(dict(card))
    result["content_hash"] = experience_content_hash(result)
    return validate_experience_card(result)


def validate_experience_card(card: object) -> dict:
    value = dict(_require_mapping(card, "experience"))
    required = {
        "schema_version",
        "experience_id",
        "revision",
        "status",
        "type",
        "phase",
        "trigger",
        "guidance",
        "anti_patterns",
        "verification_checks",
        "scope",
        "evidence",
        "provenance",
        "supersedes",
        "content_hash",
    }
    missing = required - set(value)
    if missing:
        raise ExperienceContractError(
            "experience is missing: " + ", ".join(sorted(missing))
        )
    unexpected = set(value) - required
    if unexpected:
        raise ExperienceContractError(
            "experience has unexpected fields: " + ", ".join(sorted(unexpected))
        )
    if value["schema_version"] != EXPERIENCE_CARD_VERSION:
        raise ExperienceContractError("unsupported experience schema_version")
    experience_id = _require_nonempty_text(
        value["experience_id"], "experience.experience_id", maximum=100
    )
    if not _EXPERIENCE_ID.fullmatch(experience_id):
        raise ExperienceContractError("experience_id has an invalid format")
    if not isinstance(value["revision"], int) or isinstance(value["revision"], bool):
        raise ExperienceContractError("experience.revision must be an integer")
    if value["revision"] < 1:
        raise ExperienceContractError("experience.revision must be positive")
    if value["status"] not in CARD_STATUSES:
        raise ExperienceContractError("experience.status is unsupported")
    if value["type"] not in EXPERIENCE_TYPES:
        raise ExperienceContractError("experience.type is unsupported")
    if value["phase"] not in DECISION_PHASES:
        raise ExperienceContractError("experience.phase is unsupported")

    trigger = dict(_require_mapping(value["trigger"], "experience.trigger"))
    trigger_fields = {
        "events",
        "page_types",
        "required_constraint_tags",
        "predicates",
    }
    if set(trigger) != trigger_fields:
        raise ExperienceContractError(
            "experience.trigger must contain exactly: "
            + ", ".join(sorted(trigger_fields))
        )
    events = _text_list(
        trigger.get("events"),
        "experience.trigger.events",
        maximum_items=len(RECALL_EVENTS),
        allow_empty=False,
        item_maximum=64,
    )
    if not set(events) <= RECALL_EVENTS:
        raise ExperienceContractError("experience.trigger.events contains unknown values")
    if any(value["phase"] not in RECALL_EVENT_PHASES[event] for event in events):
        raise ExperienceContractError(
            "experience phase is incompatible with one or more recall events"
        )
    page_types = _text_list(
        trigger.get("page_types", []),
        "experience.trigger.page_types",
        maximum_items=8,
        item_maximum=64,
    )
    if not set(page_types) <= PAGE_TYPES:
        raise ExperienceContractError(
            "experience.trigger.page_types contains unknown values"
        )
    constraint_tags = _text_list(
        trigger.get("required_constraint_tags", []),
        "experience.trigger.required_constraint_tags",
        maximum_items=16,
        item_maximum=64,
    )
    predicates = _text_list(
        trigger.get("predicates", []),
        "experience.trigger.predicates",
        maximum_items=len(TRIGGER_PREDICATES),
        item_maximum=64,
    )
    if not set(predicates) <= TRIGGER_PREDICATES:
        raise ExperienceContractError("experience.trigger.predicates contains unknown values")
    trigger.update(
        {
            "events": events,
            "page_types": page_types,
            "required_constraint_tags": constraint_tags,
            "predicates": predicates,
        }
    )
    value["trigger"] = trigger

    value["guidance"] = _text_list(
        value["guidance"],
        "experience.guidance",
        maximum_items=3,
        allow_empty=False,
    )
    value["anti_patterns"] = _text_list(
        value["anti_patterns"],
        "experience.anti_patterns",
        maximum_items=3,
    )
    value["verification_checks"] = _text_list(
        value["verification_checks"],
        "experience.verification_checks",
        maximum_items=3,
    )

    scope = dict(_require_mapping(value["scope"], "experience.scope"))
    scope_fields = {*SCOPE_CONTRACT, "categories"}
    if set(scope) != scope_fields:
        raise ExperienceContractError(
            "experience.scope must contain exactly: "
            + ", ".join(sorted(scope_fields))
        )
    for name, expected in SCOPE_CONTRACT.items():
        if scope.get(name) != expected:
            raise ExperienceContractError(
                f"experience.scope.{name} must equal {expected!r}"
            )
    scope["categories"] = _text_list(
        scope.get("categories", []),
        "experience.scope.categories",
        maximum_items=32,
        item_maximum=100,
    )
    value["scope"] = scope

    evidence = dict(_require_mapping(value["evidence"], "experience.evidence"))
    evidence_fields = {
        "source_kind",
        "supporting_trajectory_ids",
        "contradicting_trajectory_ids",
        "support_count",
    }
    if set(evidence) != evidence_fields:
        raise ExperienceContractError(
            "experience.evidence must contain exactly: "
            + ", ".join(sorted(evidence_fields))
        )
    if evidence.get("source_kind") not in SOURCE_KINDS:
        raise ExperienceContractError("experience.evidence.source_kind is unsupported")
    supporting = _text_list(
        evidence.get("supporting_trajectory_ids", []),
        "experience.evidence.supporting_trajectory_ids",
        maximum_items=256,
        item_maximum=128,
    )
    contradicting = _text_list(
        evidence.get("contradicting_trajectory_ids", []),
        "experience.evidence.contradicting_trajectory_ids",
        maximum_items=256,
        item_maximum=128,
    )
    if set(supporting) & set(contradicting):
        raise ExperienceContractError(
            "supporting and contradicting trajectory IDs must be disjoint"
        )
    support_count = evidence.get("support_count")
    if (
        not isinstance(support_count, int)
        or isinstance(support_count, bool)
        or support_count != len(supporting)
    ):
        raise ExperienceContractError(
            "experience.evidence.support_count must equal the supporting ID count"
        )
    evidence.update(
        {
            "supporting_trajectory_ids": supporting,
            "contradicting_trajectory_ids": contradicting,
            "support_count": support_count,
        }
    )
    value["evidence"] = evidence

    provenance = dict(_require_mapping(value["provenance"], "experience.provenance"))
    provenance_fields = {"extractor", "extractor_revision", "prompt_version", "created_at"}
    if set(provenance) != provenance_fields:
        raise ExperienceContractError(
            "experience.provenance must contain exactly: "
            + ", ".join(sorted(provenance_fields))
        )
    for field in ("extractor", "extractor_revision", "prompt_version", "created_at"):
        provenance[field] = _require_nonempty_text(
            provenance.get(field), f"experience.provenance.{field}", maximum=256
        )
    value["provenance"] = provenance
    if value["supersedes"] is not None:
        _require_nonempty_text(value["supersedes"], "experience.supersedes", maximum=128)

    semantic_text = canonical_json(experience_semantic_content(value))
    if _PRODUCT_ID.search(semantic_text):
        raise ExperienceContractError("experience semantic content contains a product ID")
    if _EXPLICIT_PRICE.search(semantic_text):
        raise ExperienceContractError("experience semantic content contains a task-specific price")
    folded = semantic_text.casefold()
    leaked = [term for term in _FORBIDDEN_SEMANTIC_TERMS if term in folded]
    if leaked:
        raise ExperienceContractError(
            "experience semantic content contains forbidden fields: "
            + ", ".join(leaked)
        )
    expected_hash = experience_content_hash(value)
    if value["content_hash"] != expected_hash:
        raise ExperienceContractError("experience.content_hash does not match semantic content")
    return value


@dataclass(frozen=True)
class ExperienceStateView:
    task_id: int
    public_query: str
    constraint_tags: tuple[str, ...]
    phase: str
    page_type: str
    current_product_id: str | None
    seen_product_ids: tuple[str, ...]
    candidate_set_hash: str
    unverified_constraints: tuple[str, ...]
    selected_options: tuple[str, ...]
    last_tool_name: str | None
    last_guard_reason: str | None
    repeat_action_count: int
    executed_steps: int
    remaining_steps: int
    predicates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.phase not in DECISION_PHASES:
            raise ExperienceContractError(f"unsupported decision phase: {self.phase!r}")
        if min(self.executed_steps, self.remaining_steps, self.repeat_action_count) < 0:
            raise ExperienceContractError("experience state counts cannot be negative")
        if not set(self.predicates) <= TRIGGER_PREDICATES:
            raise ExperienceContractError("experience state contains unknown predicates")

    def to_dict(self) -> dict:
        return asdict(self)

    def signature(self, event: str) -> str:
        if event not in RECALL_EVENTS:
            raise ExperienceContractError(f"unsupported recall event: {event!r}")
        return canonical_sha256({"event": event, "state": self.to_dict()})


@dataclass(frozen=True)
class ExperienceBundle:
    recall_event: str
    state_signature_hash: str
    eligible_experience_ids: tuple[str, ...]
    cards: tuple[dict, ...]
    scores: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.recall_event not in RECALL_EVENTS:
            raise ExperienceContractError("bundle recall_event is unsupported")
        selected = [card.get("experience_id") for card in self.cards]
        if len(selected) != len(set(selected)):
            raise ExperienceContractError("bundle contains duplicate experience IDs")
        if not set(selected) <= set(self.eligible_experience_ids):
            raise ExperienceContractError(
                "bundle selected experiences must be a subset of eligible IDs"
            )
        for experience_id in selected:
            score = self.scores.get(experience_id)
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise ExperienceContractError(
                    f"bundle score is missing for {experience_id!r}"
                )

    @property
    def selected_experience_ids(self) -> tuple[str, ...]:
        return tuple(str(card["experience_id"]) for card in self.cards)
