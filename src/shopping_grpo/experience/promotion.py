"""Fail-closed promotion of validated candidate experiences into a frozen store."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import re

from shopping_grpo.experience.contracts import (
    ExperienceContractError,
    finalize_experience_card,
    validate_experience_card,
)


PROMOTION_DECISION_VERSION = "shopping-experience-promotion-decision-v1"


def validate_promotion_decision(decision: object) -> dict:
    if not isinstance(decision, Mapping):
        raise ExperienceContractError("promotion decision must be an object")
    value = deepcopy(dict(decision))
    required = {
        "schema_version",
        "experience_id",
        "revision",
        "candidate_validation_passed",
        "store_regression_passed",
        "validation_manifest_hash",
        "decision",
        "reason",
    }
    missing = required - set(value)
    if missing:
        raise ExperienceContractError(
            "promotion decision is missing: " + ", ".join(sorted(missing))
        )
    unexpected = set(value) - required
    if unexpected:
        raise ExperienceContractError(
            "promotion decision has unexpected fields: "
            + ", ".join(sorted(unexpected))
        )
    if value["schema_version"] != PROMOTION_DECISION_VERSION:
        raise ExperienceContractError("unsupported promotion decision version")
    if value["decision"] not in {"promote", "reject"}:
        raise ExperienceContractError("promotion decision must be promote or reject")
    for field in ("candidate_validation_passed", "store_regression_passed"):
        if not isinstance(value[field], bool):
            raise ExperienceContractError(f"promotion decision {field} must be boolean")
    if value["decision"] == "promote" and not (
        value["candidate_validation_passed"] and value["store_regression_passed"]
    ):
        raise ExperienceContractError(
            "experience cannot be promoted unless both validation gates passed"
        )
    if not isinstance(value["revision"], int) or value["revision"] < 1:
        raise ExperienceContractError("promotion decision revision is invalid")
    for field in ("experience_id", "validation_manifest_hash", "reason"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ExperienceContractError(f"promotion decision {field} is required")
    if not re.fullmatch(r"[0-9a-f]{64}", value["validation_manifest_hash"]):
        raise ExperienceContractError("validation_manifest_hash must be SHA-256")
    return value


def apply_promotion_decisions(
    cards: Iterable[Mapping],
    decisions: Iterable[Mapping],
    *,
    expected_validation_manifest_hash: str | None = None,
) -> list[dict]:
    validated_cards = [validate_experience_card(card) for card in cards]
    if any(card["status"] not in {"candidate", "validated"} for card in validated_cards):
        raise ExperienceContractError(
            "promotion input cards must have candidate or validated status"
        )
    decision_by_key = {}
    for raw_decision in decisions:
        decision = validate_promotion_decision(raw_decision)
        if (
            expected_validation_manifest_hash is not None
            and decision["validation_manifest_hash"]
            != expected_validation_manifest_hash
        ):
            raise ExperienceContractError(
                "promotion decision references a different validation manifest"
            )
        key = (decision["experience_id"], decision["revision"])
        if key in decision_by_key:
            raise ExperienceContractError("duplicate promotion decision")
        decision_by_key[key] = decision
    output = []
    for card in validated_cards:
        key = (card["experience_id"], card["revision"])
        decision = decision_by_key.get(key)
        if decision is None:
            continue
        promoted = deepcopy(card)
        promoted["status"] = "active" if decision["decision"] == "promote" else "rejected"
        output.append(finalize_experience_card(promoted))
    unused = set(decision_by_key) - {
        (card["experience_id"], card["revision"]) for card in validated_cards
    }
    if unused:
        raise ExperienceContractError("promotion decisions reference unknown experiences")
    return sorted(output, key=lambda card: (card["experience_id"], card["revision"]))
