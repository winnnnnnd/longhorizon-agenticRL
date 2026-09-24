"""Pure, reproducible goal helpers for Environment v2."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re


CONSTRAINT_CONTRACT_VERSION = "shopping-task-constraints-v1"
EVIDENCE_CONSTRAINT_CONTRACT_VERSION = "shopping-evidence-constraints-v1"


def _clean_annotation_values(value):
    """Normalize task-authored annotations without inferring new constraints."""
    if not isinstance(value, list):
        return [], False
    cleaned = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned, True


def compile_task_constraint_contract(instruction_record):
    """Compile the task's existing annotations into a fail-closed contract.

    Environment v2 uses only fields already attached to the current natural
    language instruction. It deliberately does not infer brand/model or read
    extra attributes from the target product.
    """
    if not isinstance(instruction_record, dict):
        instruction_record = {}
    attributes, attributes_valid = _clean_annotation_values(
        instruction_record.get("attributes")
    )
    options, options_valid = _clean_annotation_values(
        instruction_record.get("instruction_options")
    )
    instruction_valid = bool(
        re.sub(r"\s+", "", str(instruction_record.get("instruction") or ""))
    )
    complete = instruction_valid and attributes_valid and options_valid
    return {
        "hard_constraints": {
            "complete": complete,
            "contract_version": CONSTRAINT_CONTRACT_VERSION,
            "annotation_source": "instruction.attributes",
            "core_functions": attributes,
            "brand": [],
            "model": [],
            "key_specs": [],
            # Options continue to be checked by Reward v2's key_options gate.
            "annotated_option_count": len(options),
        },
        # The current task data does not label hard versus soft preferences.
        # Do not guess that distinction from keywords.
        "weighted_preferences": [],
    }


def compile_evidence_constraint_contract(goal):
    """Adapt existing Reward v3 features for trajectory evidence tracking.

    This contract is runtime metadata and is not part of the model-visible
    observation.  It contains only the user instruction and the already
    compiled Reward v3 requirement fields; target ASIN/title are never copied.
    """

    goal = goal if isinstance(goal, dict) else {}
    rows = []

    def add(constraint_type, attribute, operator, expected_value, description=""):
        identity = {
            "constraint_type": str(constraint_type),
            "attribute": str(attribute),
            "operator": str(operator),
            "expected_value": expected_value,
        }
        digest = hashlib.sha256(
            json.dumps(
                identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "constraint_id": digest,
                **identity,
                "description": str(description or ""),
            }
        )

    if goal.get("category"):
        add(
            "category",
            "product.category",
            "in_category",
            goal["category"],
            "商品品类符合用户请求",
        )
    for value in goal.get("expected_brand") or []:
        add("brand", "product.brand", "eq", value, f"品牌为{value}")
    for value in goal.get("expected_model") or []:
        add("model", "product.model", "eq", value, f"型号为{value}")
    for value in goal.get("expected_core_functions") or []:
        add(
            "core_function",
            "product.key_attributes",
            "contains",
            value,
            f"商品支持{value}",
        )
    required_options = goal.get("required_options_by_key")
    if isinstance(required_options, dict):
        for axis, requirement in required_options.items():
            add(
                "option",
                f"product.available_options.{axis}",
                "eq",
                requirement,
                f"规格{axis}满足要求",
            )
    for unresolved in goal.get("unresolved_option_requirements") or []:
        if isinstance(unresolved, dict) and unresolved.get("value"):
            add(
                "option",
                "product.available_options.unresolved",
                "contains",
                unresolved["value"],
                f"规格包含{unresolved['value']}",
            )
    if goal.get("price_upper") is not None:
        add(
            "budget_upper",
            "product.price",
            "lte",
            {"value": goal["price_upper"], "currency": "CNY"},
            f"价格不超过{goal['price_upper']}元",
        )
    return {
        "version": EVIDENCE_CONSTRAINT_CONTRACT_VERSION,
        "query": str(goal.get("instruction_text") or ""),
        "constraints": rows,
    }


def explicit_budget_from_instruction(instruction):
    """Extract a clearly stated upper budget; return None when ambiguous."""
    text = str(instruction or "").replace(",", "")

    def scaled(number, unit):
        value = float(number)
        normalized_unit = str(unit or "").casefold()
        if normalized_unit == "万":
            value *= 10000
        elif normalized_unit in {"千", "k"}:
            value *= 1000
        return value

    shorthand = re.search(
        r"预算(?:控制)?在?\s*(\d+)\s*万\s*(\d+)\s*(?:千)?\s*(以内|以下|内|左右)?",
        text,
    )
    if shorthand:
        value = float(shorthand.group(1)) * 10000 + float(shorthand.group(2)) * 1000
        if shorthand.group(3) == "左右":
            value *= 1.1
        return value

    # Ranges are explicit upper bounds. Parse them before single-value
    # patterns so “10-20” is never mistaken for an upper bound of 10.
    price_range = re.search(
        r"(?:预算|价格)(?:控制)?在?\s*"
        r"(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?\s*"
        r"(?:-|~|～|至|到)\s*"
        r"(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?"
        r"(?:之间|以内|以下|左右)?",
        text,
    )
    if price_range:
        low = scaled(price_range.group(1), price_range.group(2))
        high = scaled(price_range.group(3), price_range.group(4))
        if low > 0 and high >= low:
            return high

    # “4k+” describes a lower/open-ended price, not a maximum.
    if re.search(
        r"(?:预算|价格)(?:控制)?在?\s*\d+(?:\.\d+)?\s*[kK]\s*\+",
        text,
    ):
        return None

    patterns = (
        r"预算(?:控制)?在?\s*(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?(以内|以下|内|左右)?(?!\s*[-~～至到+kK])",
        r"价格(?:控制)?在?\s*(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?(以内|以下|内|左右)?(?!\s*[-~～至到+kK])",
        r"(?:不超过|不高于|最高)\s*(\d+(?:\.\d+)?)\s*(万|千)?\s*元",
        r"(\d+(?:\.\d+)?)\s*(万|千)?\s*元(以内|以下)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = scaled(match.group(1), match.group(2))
            qualifier = match.group(3) if match.lastindex and match.lastindex >= 3 else None
            # “左右” is not a strict upper bound. Environment v2 freezes a
            # small deterministic tolerance rather than letting target items
            # just above the round-number budget become impossible.
            if qualifier == "左右":
                value *= 1.1
            if value > 0:
                return value
    return None


def _price_range_above(price):
    if price <= 100:
        step = 3
    elif price <= 1000:
        step = 10
    elif price <= 5000:
        step = 50
    elif price <= 10000:
        step = 100
    else:
        step = 4
    base = math.ceil(price / 10) * 10
    return [base + index * 10 for index in range(step)]


def deterministic_price_upper(asin, instruction, price):
    explicit = explicit_budget_from_instruction(instruction)
    if explicit is not None:
        return explicit
    price_range = _price_range_above(float(price))
    if len(price_range) < 2:
        return 10000000
    digest = hashlib.sha256(
        f"{asin}\0{instruction}".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    _, upper = sorted(rng.sample(price_range, 2))
    return upper
