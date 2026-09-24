"""Trajectory-local, deterministic evidence tracking for shopping rollouts.

The store in this module is deliberately made only from JSON-compatible
objects.  It can therefore be copied into Ray results, JSONL diagnostics and
offline replay inputs without retaining a Python object graph.  It is runtime
metadata: callers must not append it to the Actor-visible observation.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import re
import unicodedata


EVIDENCE_STORE_VERSION = "shopping-evidence-store-v1"
EVIDENCE_DELTA_VERSION = "shopping-evidence-delta-v1"
UNKNOWN = "unknown"
VERIFIED_PASS = "verified_pass"
VERIFIED_FAIL = "verified_fail"
CONFLICT = "conflict"
CONSTRAINT_STATUSES = {UNKNOWN, VERIFIED_PASS, VERIFIED_FAIL, CONFLICT}
SEMANTIC_RESULT_OVERLAP = 0.80


def canonical_json(value: object) -> str:
    """Return the canonical JSON representation used by stable hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


def _normalized_text(value: object, *, remove_space: bool = False) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    if remove_space:
        return re.sub(r"\s+", "", text)
    return re.sub(r"\s+", " ", text)


def _normalized_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            _normalized_text(key): _normalized_json_value(nested)
            for key, nested in value.items()
            if nested is not None
        }
    if isinstance(value, (list, tuple)):
        return [_normalized_json_value(nested) for nested in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value).casefold()
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        return _normalized_text(value)
    return _normalized_text(value)


def normalized_action(tool_name: object, arguments: object) -> dict:
    """Normalize a tool call independently of dict order and text casing."""

    normalized_name = _normalized_text(tool_name, remove_space=True)
    raw_arguments = arguments if isinstance(arguments, Mapping) else {}
    return {
        "tool": normalized_name,
        "arguments": _normalized_json_value(raw_arguments),
    }


def action_hash(tool_name: object, arguments: object) -> str:
    """Hash ``normalized_tool_name + canonical_json(normalized_args)``."""

    normalized = normalized_action(tool_name, arguments)
    payload = normalized["tool"] + canonical_json(normalized["arguments"])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def result_hash(observation_state: object) -> str:
    """Return a stable hash of a structured tool result."""

    payload = observation_state if isinstance(observation_state, Mapping) else {}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def evidence_id(
    product_id: object,
    variant_id: object,
    attribute: object,
    normalized_value: object,
) -> str:
    """Hash product + variant + attribute + normalized value exactly once."""

    parts = (
        _normalized_text(product_id, remove_space=True),
        _normalized_text(variant_id, remove_space=True),
        _normalized_text(attribute, remove_space=True),
        canonical_json(_normalized_json_value(normalized_value)),
    )
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def create_evidence_store(
    *,
    task_id: int,
    requirement_text: object = "",
    constraint_contract: object = None,
    goal_options: object = None,
) -> dict:
    """Create one JSON-compatible store for exactly one trajectory."""

    constraints = _normalize_constraints(
        constraint_contract,
        goal_options=goal_options,
    )
    store = {
        "version": EVIDENCE_STORE_VERSION,
        "task_id": int(task_id),
        "requirement_text": str(requirement_text or ""),
        "constraints": {item["constraint_id"]: item for item in constraints},
        "relevant_product_ids": [],
        "candidates": {},
        "facts": {},
        "steps": [],
        "consecutive_no_progress_steps": 0,
        "progress_step_count": 0,
        "no_progress_repeat_count": 0,
        "semantic_repeat_count": 0,
        "constraint_coverage": {},
    }
    store["constraint_coverage"] = constraint_coverage(store)
    return store


def _constraint_identity(item: Mapping) -> str:
    payload = {
        "constraint_type": str(item.get("constraint_type") or "generic"),
        "attribute": str(item.get("attribute") or item.get("field_path") or ""),
        "operator": str(item.get("operator") or "contains"),
        "expected_value": item.get("expected_value"),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _constraint_record(item: Mapping) -> dict:
    identifier = str(
        item.get("constraint_id")
        or item.get("rubric_id")
        or item.get("candidate_id")
        or _constraint_identity(item)
    )
    attribute = str(item.get("attribute") or item.get("field_path") or "")
    return {
        "constraint_id": identifier,
        "constraint_type": str(item.get("constraint_type") or "generic"),
        "attribute": attribute,
        "operator": str(item.get("operator") or "contains"),
        "expected_value": deepcopy(item.get("expected_value")),
        "description": str(item.get("description") or item.get("description_hint") or ""),
        "status": UNKNOWN,
        "product_statuses": {},
        "evidence_ids": [],
        "conflict_count": 0,
    }


def _normalize_constraints(contract: object, *, goal_options: object) -> list[dict]:
    raw = contract if isinstance(contract, Mapping) else {}
    rows = raw.get("constraints")
    if not isinstance(rows, list):
        rows = raw.get("rubrics")
    rows = list(rows) if isinstance(rows, list) else []

    hard = raw.get("hard_constraints")
    if isinstance(hard, Mapping):
        for value in hard.get("core_functions") or []:
            rows.append(
                {
                    "constraint_type": "core_function",
                    "attribute": "product.key_attributes",
                    "operator": "contains",
                    "expected_value": value,
                }
            )
        for constraint_type in ("brand", "model", "key_specs"):
            for value in hard.get(constraint_type) or []:
                rows.append(
                    {
                        "constraint_type": constraint_type,
                        "attribute": f"product.{constraint_type}",
                        "operator": "contains",
                        "expected_value": value,
                    }
                )

    has_option_constraint = any(
        str(row.get("constraint_type")) == "option"
        for row in rows
        if isinstance(row, Mapping)
    )
    if not has_option_constraint:
        for value in goal_options if isinstance(goal_options, list) else []:
            rows.append(
                {
                    "constraint_type": "option",
                    "attribute": "product.available_options",
                    "operator": "contains",
                    "expected_value": value,
                }
            )

    normalized = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        record = _constraint_record(row)
        identity = _constraint_identity(record)
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(record)
    return normalized


def record_tool_evidence(
    store: dict,
    *,
    tool_name: object,
    arguments: object,
    observation_state: object,
    step_index: int,
    environment_action: object = None,
) -> dict:
    """Extract evidence, update constraint state and return step log fields."""

    if store.get("version") != EVIDENCE_STORE_VERSION:
        raise ValueError("unsupported evidence store version")
    structured = observation_state if isinstance(observation_state, Mapping) else {}
    current_action_hash = action_hash(tool_name, arguments)
    current_result_hash = result_hash(structured)
    result_products = _result_products(structured)
    result_product_ids = [product_id for product_id, _ in result_products]
    source = {
        "step": int(step_index),
        "tool": str(tool_name),
        "action": str(environment_action or ""),
        "action_hash": current_action_hash,
        "result_hash": current_result_hash,
    }
    delta = {
        "version": EVIDENCE_DELTA_VERSION,
        "new_relevant_candidates": [],
        "new_facts": [],
        "newly_verified_constraints": [],
        "newly_failed_constraints": [],
        "resolved_conflicts": [],
        "has_progress": False,
    }
    page_type = str(structured.get("page_type") or "")
    detailed_page = page_type in {"product_detail", "information_subpage"}
    relevant_ids = store["relevant_product_ids"]

    for product_id, product in result_products:
        relevant = product_id in relevant_ids or _product_is_relevant(store, product)
        if relevant and product_id not in relevant_ids:
            relevant_ids.append(product_id)
            delta["new_relevant_candidates"].append(product_id)
        if relevant:
            _merge_candidate(store, product_id, product, source)
        if not detailed_page:
            continue
        facts = _extract_product_facts(
            store,
            product_id=product_id,
            product=product,
            structured=structured,
        )
        newly_added = []
        for fact in facts:
            identifier, is_new = _merge_fact(store, fact, source)
            if is_new:
                newly_added.append((identifier, fact))
                if relevant and _fact_is_progress_relevant(store, fact):
                    delta["new_facts"].append(identifier)
        if relevant:
            _update_constraints(
                store,
                product_id=product_id,
                product=product,
                structured=structured,
                new_facts=newly_added,
                step_index=int(step_index),
                delta=delta,
            )

    delta["has_progress"] = any(
        delta[key]
        for key in (
            "new_relevant_candidates",
            "new_facts",
            "newly_verified_constraints",
            "newly_failed_constraints",
            "resolved_conflicts",
        )
    )
    repeat_type = _repeat_type(
        store,
        tool_name=str(tool_name),
        action_hash_value=current_action_hash,
        result_hash_value=current_result_hash,
        product_ids=result_product_ids,
        has_progress=bool(delta["has_progress"]),
    )
    if delta["has_progress"]:
        store["progress_step_count"] += 1
        store["consecutive_no_progress_steps"] = 0
    else:
        store["consecutive_no_progress_steps"] += 1
    if repeat_type == "no_progress_repeat":
        store["no_progress_repeat_count"] += 1
    elif repeat_type == "semantic_repeat":
        store["semantic_repeat_count"] += 1

    coverage = constraint_coverage(store)
    store["constraint_coverage"] = coverage
    step_record = {
        "step": int(step_index),
        "tool": str(tool_name),
        "action_hash": current_action_hash,
        "result_hash": current_result_hash,
        "result_product_ids": result_product_ids,
        "evidence_delta": delta,
        "has_progress": bool(delta["has_progress"]),
        "repeat_type": repeat_type,
        "consecutive_no_progress_steps": int(
            store["consecutive_no_progress_steps"]
        ),
        "constraint_coverage": deepcopy(coverage),
    }
    store["steps"].append(step_record)
    return deepcopy(step_record)


def record_empty_tool_evidence(
    store: dict,
    *,
    tool_name: object,
    arguments: object,
    step_index: int,
) -> dict:
    """Record a non-environment tool step without fabricating evidence."""

    return record_tool_evidence(
        store,
        tool_name=tool_name,
        arguments=arguments,
        observation_state={},
        step_index=step_index,
    )


def constraint_coverage(store: Mapping) -> dict:
    constraints = store.get("constraints")
    values = list(constraints.values()) if isinstance(constraints, Mapping) else []
    counts = {status: 0 for status in (VERIFIED_PASS, VERIFIED_FAIL, UNKNOWN, CONFLICT)}
    statuses = {}
    for constraint in values:
        status = str(constraint.get("status") or UNKNOWN)
        if status not in CONSTRAINT_STATUSES:
            status = UNKNOWN
        counts[status] += 1
        statuses[str(constraint.get("constraint_id"))] = status
    verified = counts[VERIFIED_PASS] + counts[VERIFIED_FAIL]
    return {
        "total": len(values),
        "verified": verified,
        "coverage": verified / len(values) if values else 0.0,
        "counts": counts,
        "statuses": statuses,
    }


def classify_timeout(
    store: Mapping,
    *,
    no_progress_window: int = 4,
) -> str:
    """Classify a genuine step-budget exhaustion from evidence history."""

    consecutive = int(store.get("consecutive_no_progress_steps", 0))
    repeats = int(store.get("no_progress_repeat_count", 0)) + int(
        store.get("semantic_repeat_count", 0)
    )
    if consecutive >= int(no_progress_window) or repeats >= int(no_progress_window):
        return "no_progress_timeout"
    return "productive_timeout"


def _result_products(structured: Mapping) -> list[tuple[str, dict]]:
    page_type = str(structured.get("page_type") or "")
    if page_type == "search_results":
        raw_products = structured.get("products")
        products = raw_products if isinstance(raw_products, list) else []
    elif page_type in {"product_detail", "information_subpage"}:
        raw_product = structured.get("product")
        products = [raw_product] if isinstance(raw_product, Mapping) else []
    else:
        products = []
    result = []
    for product in products:
        if not isinstance(product, Mapping):
            continue
        product_id = _normalized_text(product.get("asin"), remove_space=True)
        if product_id:
            result.append((product_id, dict(product)))
    return result


def _flatten_text(value: object) -> str:
    if isinstance(value, Mapping):
        values = []
        for key, nested in value.items():
            values.extend((str(key), _flatten_text(nested)))
        return " ".join(values)
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_text(item) for item in value)
    return str(value or "")


def _category_match(required: object, actual: object) -> bool:
    required_leaf = _normalized_text(str(required or "").split("›")[-1], remove_space=True)
    actual_leaf = _normalized_text(str(actual or "").split("›")[-1], remove_space=True)
    return bool(required_leaf and actual_leaf and required_leaf == actual_leaf)


def _product_is_relevant(store: Mapping, product: Mapping) -> bool:
    constraints = store.get("constraints") or {}
    product_text = _normalized_text(_flatten_text(product), remove_space=True)
    category_constraints = [
        constraint
        for constraint in constraints.values()
        if str(constraint.get("constraint_type") or "") == "category"
    ]
    if category_constraints and product.get("category"):
        return any(
            _category_match(
                constraint.get("expected_value"),
                product.get("category"),
            )
            for constraint in category_constraints
        )
    decisive = False
    for constraint in constraints.values():
        constraint_type = str(constraint.get("constraint_type") or "")
        expected = constraint.get("expected_value")
        if constraint_type in {"brand", "model", "core_function", "key_specs"}:
            decisive = True
            expected_text = _normalized_text(expected, remove_space=True)
            if expected_text and expected_text in product_text:
                return True
    if decisive:
        return False
    requirement = _normalized_text(store.get("requirement_text"), remove_space=True)
    title = _normalized_text(product.get("title"), remove_space=True)
    return bool(title and requirement and (title in requirement or requirement in title))


def _variant_id(structured: Mapping) -> str:
    selected = structured.get("selected_options")
    if not isinstance(selected, Mapping) or not selected:
        return ""
    return canonical_json(_normalized_json_value(selected))


def _fact(product_id: str, variant: str, attribute: str, value: object) -> dict:
    normalized = _normalized_json_value(value)
    return {
        "product_id": product_id,
        "variant_id": variant,
        "attribute": str(attribute),
        "value": deepcopy(value),
        "normalized_value": normalized,
        "evidence_id": evidence_id(product_id, variant, attribute, normalized),
    }


def _extract_product_facts(
    store: Mapping,
    *,
    product_id: str,
    product: Mapping,
    structured: Mapping,
) -> list[dict]:
    variant = _variant_id(structured)
    facts = []
    for attribute in (
        "title",
        "brand",
        "category",
        "price",
        "inventory",
        "stock",
        "availability",
        "availability_status",
        "delivery",
        "delivery_time",
        "shipping",
    ):
        value = (
            structured.get("selected_price")
            if attribute == "price" and structured.get("selected_price") is not None
            else product.get(attribute)
        )
        if value not in (None, "", [], {}):
            facts.append(_fact(product_id, variant, attribute, value))

    attributes = product.get("key_attributes")
    if isinstance(attributes, Mapping):
        for key, value in attributes.items():
            facts.append(_fact(product_id, variant, f"attribute:{key}", value))
    elif isinstance(attributes, list):
        for value in attributes:
            text = str(value or "").strip()
            if not text:
                continue
            match = re.match(r"^([^:=：]{1,40})\s*[:=：]\s*(.+)$", text)
            attribute = f"attribute:{match.group(1)}" if match else "key_attribute"
            fact_value = match.group(2) if match else text
            facts.append(_fact(product_id, variant, attribute, fact_value))

    available = structured.get("available_options")
    if isinstance(available, Mapping):
        for axis, values in available.items():
            normalized_values = values if isinstance(values, list) else [values]
            normalized_values = sorted(
                normalized_values,
                key=lambda value: _normalized_text(value, remove_space=True),
            )
            facts.append(
                _fact(
                    product_id,
                    variant,
                    f"available_options:{axis}",
                    normalized_values,
                )
            )
            for value in normalized_values:
                facts.append(
                    _fact(
                        product_id,
                        variant,
                        f"option_available:{axis}:{value}",
                        True,
                    )
                )

    selected = structured.get("selected_options")
    if isinstance(selected, Mapping):
        for axis, value in selected.items():
            facts.append(_fact(product_id, variant, f"selected_option:{axis}", value))

    # A missing required value is explicit negative evidence only when the
    # corresponding option axis is present in the structured result.
    for constraint in (store.get("constraints") or {}).values():
        if str(constraint.get("constraint_type")) != "option":
            continue
        expected, expected_axis = _option_requirement(constraint)
        matched_axis, values = _available_option_values(available, expected_axis)
        if matched_axis is None:
            continue
        present = _normalized_text(expected, remove_space=True) in {
            _normalized_text(value, remove_space=True) for value in values
        }
        facts.append(
            _fact(
                product_id,
                variant,
                f"required_option_available:{matched_axis}:{expected}",
                present,
            )
        )

    if structured.get("content") not in (None, ""):
        facts.append(
            _fact(
                product_id,
                variant,
                f"subpage:{structured.get('subpage') or 'information'}",
                structured.get("content"),
            )
        )
    return facts


def _merge_fact(store: dict, fact: dict, source: dict) -> tuple[str, bool]:
    identifier = fact["evidence_id"]
    existing = store["facts"].get(identifier)
    if existing is None:
        existing = {**deepcopy(fact), "sources": []}
        store["facts"][identifier] = existing
        is_new = True
    else:
        is_new = False
    source_key = (source["step"], source["action_hash"], source["result_hash"])
    if source_key not in {
        (item["step"], item["action_hash"], item["result_hash"])
        for item in existing["sources"]
    }:
        existing["sources"].append(deepcopy(source))
    return identifier, is_new


def _merge_candidate(
    store: dict,
    product_id: str,
    product: Mapping,
    source: Mapping,
) -> None:
    candidate = store["candidates"].setdefault(
        product_id,
        {
            "product_id": product_id,
            "title": str(product.get("title") or ""),
            "brand": str(product.get("brand") or ""),
            "category": str(product.get("category") or ""),
            "sources": [],
        },
    )
    source_key = (source["step"], source["action_hash"], source["result_hash"])
    if source_key not in {
        (item["step"], item["action_hash"], item["result_hash"])
        for item in candidate["sources"]
    }:
        candidate["sources"].append(deepcopy(dict(source)))


def _fact_is_progress_relevant(store: Mapping, fact: Mapping) -> bool:
    attribute = _normalized_text(fact.get("attribute"), remove_space=True)
    requirement = _normalized_text(store.get("requirement_text"), remove_space=True)
    # Stock/availability changes always affect whether a candidate can be
    # purchased, so they remain relevant even without an explicit stock phrase.
    if any(token in attribute for token in ("inventory", "stock", "availability")):
        return True
    for constraint in (store.get("constraints") or {}).values():
        constraint_type = str(constraint.get("constraint_type") or "")
        constraint_attribute = _normalized_text(
            constraint.get("attribute"), remove_space=True
        )
        if (
            constraint_type in {"budget_upper", "price_range", "price_lower"}
            and "price" in attribute
        ):
            return True
        if constraint_type == "category" and "category" in attribute:
            return True
        if constraint_type == "brand" and "brand" in attribute:
            return True
        if constraint_type == "model" and any(token in attribute for token in ("model", "title")):
            return True
        if constraint_type in {"core_function", "key_specs"} and any(
            token in attribute for token in ("attribute", "subpage", "title")
        ):
            return True
        if constraint_type == "option" and "option" in attribute:
            return True
        if constraint_attribute and constraint_attribute.split(".")[-1] in attribute:
            return True
    return any(
        token in requirement and token in attribute
        for token in ("库存", "stock", "inventory", "送达", "delivery", "shipping")
    )


def _fact_supports_constraint(constraint: Mapping, fact: Mapping) -> bool:
    constraint_type = str(constraint.get("constraint_type") or "")
    attribute = _normalized_text(fact.get("attribute"), remove_space=True)
    if constraint_type in {"budget_upper", "price_range", "price_lower"}:
        return "price" in attribute
    if constraint_type == "category":
        return "category" in attribute
    if constraint_type == "brand":
        return "brand" in attribute
    if constraint_type == "model":
        return "model" in attribute or "title" in attribute
    if constraint_type in {"core_function", "key_specs"}:
        return any(token in attribute for token in ("attribute", "subpage", "title"))
    if constraint_type == "option":
        return "option" in attribute
    declared = _normalized_text(constraint.get("attribute"), remove_space=True)
    return bool(declared and declared.split(".")[-1] in attribute)


def _option_requirement(constraint: Mapping) -> tuple[object, str]:
    expected = constraint.get("expected_value")
    axis = ""
    if isinstance(expected, Mapping):
        axis = str(expected.get("source_axis") or expected.get("axis") or "")
        expected = expected.get("value")
    if not axis:
        attribute = str(constraint.get("attribute") or "")
        if "." in attribute:
            axis = attribute.rsplit(".", 1)[-1]
        if axis in {"available_options", "unresolved"}:
            axis = ""
    return expected, axis


def _available_option_values(available: object, expected_axis: str) -> tuple[str | None, list]:
    if not isinstance(available, Mapping):
        return None, []
    normalized_axis = _normalized_text(expected_axis, remove_space=True)
    if normalized_axis:
        for axis, values in available.items():
            if _normalized_text(axis, remove_space=True) == normalized_axis:
                return str(axis), values if isinstance(values, list) else [values]
        return None, []
    flattened = []
    for values in available.values():
        flattened.extend(values if isinstance(values, list) else [values])
    return "*", flattened


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _evaluate_constraint(
    constraint: Mapping,
    product: Mapping,
    structured: Mapping,
) -> str:
    constraint_type = str(constraint.get("constraint_type") or "")
    expected = constraint.get("expected_value")
    product_text = _normalized_text(_flatten_text(product), remove_space=True)
    expected_text = _normalized_text(expected, remove_space=True)
    if constraint_type == "category":
        if not product.get("category"):
            return UNKNOWN
        return (
            VERIFIED_PASS
            if _category_match(expected, product.get("category"))
            else VERIFIED_FAIL
        )
    if constraint_type == "brand":
        if not product.get("brand"):
            return UNKNOWN
        actual = _normalized_text(product.get("brand"), remove_space=True)
        return (
            VERIFIED_PASS
            if expected_text == actual or expected_text in actual
            else VERIFIED_FAIL
        )
    if constraint_type == "model":
        if not product.get("title"):
            return UNKNOWN
        return VERIFIED_PASS if expected_text and expected_text in product_text else VERIFIED_FAIL
    if constraint_type in {"core_function", "key_specs"}:
        if not product_text:
            return UNKNOWN
        return VERIFIED_PASS if expected_text and expected_text in product_text else VERIFIED_FAIL
    if constraint_type == "option":
        required, axis = _option_requirement(constraint)
        matched_axis, values = _available_option_values(structured.get("available_options"), axis)
        if matched_axis is None:
            return UNKNOWN
        normalized_values = {
            _normalized_text(value, remove_space=True) for value in values
        }
        return (
            VERIFIED_PASS
            if _normalized_text(required, remove_space=True) in normalized_values
            else VERIFIED_FAIL
        )
    if constraint_type in {"budget_upper", "price_lower", "price_range"}:
        actual = _finite_number(
            structured.get("selected_price", product.get("price"))
        )
        if actual is None:
            return UNKNOWN
        if constraint_type == "budget_upper":
            limit = _finite_number(
                expected.get("value") if isinstance(expected, Mapping) else expected
            )
            if limit is None:
                return UNKNOWN
            return VERIFIED_PASS if actual <= limit else VERIFIED_FAIL
        if constraint_type == "price_lower":
            limit = _finite_number(
                expected.get("value") if isinstance(expected, Mapping) else expected
            )
            if limit is None:
                return UNKNOWN
            return VERIFIED_PASS if actual >= limit else VERIFIED_FAIL
        low = _finite_number(expected.get("min")) if isinstance(expected, Mapping) else None
        high = _finite_number(expected.get("max")) if isinstance(expected, Mapping) else None
        return UNKNOWN if low is None or high is None else (
            VERIFIED_PASS if low <= actual <= high else VERIFIED_FAIL
        )
    return UNKNOWN


def _aggregate_constraint_status(product_statuses: Mapping) -> str:
    statuses = {str(item.get("status") or UNKNOWN) for item in product_statuses.values()}
    if VERIFIED_PASS in statuses:
        return VERIFIED_PASS
    if CONFLICT in statuses:
        return CONFLICT
    if VERIFIED_FAIL in statuses:
        return VERIFIED_FAIL
    return UNKNOWN


def _update_constraints(
    store: dict,
    *,
    product_id: str,
    product: Mapping,
    structured: Mapping,
    new_facts: list[tuple[str, dict]],
    step_index: int,
    delta: dict,
) -> None:
    for constraint in store["constraints"].values():
        evaluated = _evaluate_constraint(constraint, product, structured)
        if evaluated == UNKNOWN:
            continue
        old_overall = constraint["status"]
        previous = constraint["product_statuses"].get(product_id)
        previous_status = str(previous.get("status")) if isinstance(previous, Mapping) else UNKNOWN
        resolved = False
        if previous_status in {VERIFIED_PASS, VERIFIED_FAIL} and evaluated != previous_status:
            product_status = CONFLICT
            constraint["conflict_count"] += 1
        elif previous_status == CONFLICT:
            product_status = evaluated
            resolved = True
        else:
            product_status = evaluated
        constraint["product_statuses"][product_id] = {
            "status": product_status,
            "step": int(step_index),
        }
        for identifier, fact in new_facts:
            if (
                _fact_supports_constraint(constraint, fact)
                and identifier not in constraint["evidence_ids"]
            ):
                constraint["evidence_ids"].append(identifier)
        constraint["status"] = _aggregate_constraint_status(
            constraint["product_statuses"]
        )
        new_overall = constraint["status"]
        identifier = constraint["constraint_id"]
        if resolved and identifier not in delta["resolved_conflicts"]:
            delta["resolved_conflicts"].append(identifier)
        if new_overall == VERIFIED_PASS and old_overall != VERIFIED_PASS:
            delta["newly_verified_constraints"].append(identifier)
        elif new_overall == VERIFIED_FAIL and old_overall not in {
            VERIFIED_FAIL,
            VERIFIED_PASS,
        }:
            delta["newly_failed_constraints"].append(identifier)


def _result_overlap(left: list[str], right: list[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set.intersection(right_set)) / len(left_set.union(right_set))


def _repeat_type(
    store: Mapping,
    *,
    tool_name: str,
    action_hash_value: str,
    result_hash_value: str,
    product_ids: list[str],
    has_progress: bool,
) -> str | None:
    if has_progress or not store.get("steps"):
        return None
    previous = store["steps"][-1]
    overlap = _result_overlap(product_ids, previous.get("result_product_ids") or [])
    same_result = result_hash_value == previous.get("result_hash")
    if action_hash_value == previous.get("action_hash") and (
        same_result or overlap >= SEMANTIC_RESULT_OVERLAP
    ):
        return "no_progress_repeat"
    if (
        _normalized_text(tool_name, remove_space=True) == "search_products"
        and _normalized_text(previous.get("tool"), remove_space=True) == "search_products"
        and overlap >= SEMANTIC_RESULT_OVERLAP
    ):
        return "semantic_repeat"
    return None
