"""Deterministic state views and recall-event detection for shopping rollouts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re

from shopping_grpo.environment.actions import clickable_buttons, product_ids
from shopping_grpo.environment.product_id import PRODUCT_ID_CAPTURE
from shopping_grpo.experience.contracts import ExperienceStateView
from shopping_grpo.experience.segmentation import observation_page_type, public_query


_CONSTRAINT_PATTERNS = {
    "budget": re.compile(r"(?:预算|价格|不超过|以内|以下|最多|低于|元)"),
    "brand": re.compile(r"(?:品牌|牌子|官方|旗舰)"),
    "model": re.compile(r"(?:型号|款式|系列|版本)"),
    "core_function": re.compile(r"(?:功能|支持|适合|用于|可水洗|防水|降噪|续航)"),
    "option": re.compile(r"(?:颜色|尺寸|尺码|规格|容量|数量|\d+色)"),
}
_ASIN_LINE = re.compile(rf"(?m)^asin:\s*({PRODUCT_ID_CAPTURE})\s*$")
_ALL_PRODUCT_IDS = re.compile(rf"(?<!\d)({PRODUCT_ID_CAPTURE})(?!\d)")


def constraint_tags(query: str) -> tuple[str, ...]:
    return tuple(
        name for name, pattern in _CONSTRAINT_PATTERNS.items() if pattern.search(query)
    )


def _selected_options(steps: list[Mapping]) -> tuple[str, ...]:
    values = []
    for step in steps:
        if step.get("tool_name") != "select_option":
            continue
        value = (step.get("parameters") or {}).get("value")
        if value is not None:
            values.append(str(value))
    return tuple(values)


def _seen_product_ids(steps: list[Mapping], latest_observation: str) -> tuple[str, ...]:
    seen = []
    for step in steps:
        observation = str(step.get("observation") or "")
        seen.extend(_ALL_PRODUCT_IDS.findall(observation))
        parameters = step.get("parameters") or {}
        if step.get("tool_name") == "open_product" and parameters.get("asin"):
            seen.append(str(parameters["asin"]))
    seen.extend(_ALL_PRODUCT_IDS.findall(latest_observation))
    return tuple(dict.fromkeys(seen))


def _repeat_action_count(steps: list[Mapping]) -> int:
    signatures = []
    for step in steps:
        signatures.append(
            (
                str(step.get("tool_name") or ""),
                json.dumps(
                    step.get("parameters") or {},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
    return sum(
        signature in signatures[max(0, index - 3) : index]
        for index, signature in enumerate(signatures)
    )


def _last_guard_reason(trajectory: Mapping, executed_steps: int) -> str | None:
    blocked = trajectory.get("blocked_tool_calls") or []
    if not blocked or not isinstance(blocked[-1], Mapping):
        return None
    if blocked[-1].get("step_index") != int(executed_steps):
        return None
    return str(blocked[-1].get("reason") or "unknown")


def _phase(steps: list[Mapping], guard_reason: str | None) -> str:
    if guard_reason:
        return "error_recovery"
    if not steps:
        return "task_understanding"
    tool = str(steps[-1].get("tool_name") or "")
    return {
        "search_products": "search",
        "next_page": "search",
        "open_product": "candidate_screening",
        "view_description": "detail_verification",
        "view_features": "detail_verification",
        "view_reviews": "detail_verification",
        "view_attributes": "detail_verification",
        "select_option": "option_selection",
        "buy_now": "pre_purchase",
        "finish_without_purchase": "termination",
    }.get(tool, "task_understanding")


def _search_stagnated(steps: list[Mapping]) -> bool:
    searches = [step for step in steps if step.get("tool_name") == "search_products"]
    if len(searches) < 2:
        return False
    previous, latest = searches[-2:]
    previous_query = str((previous.get("parameters") or {}).get("query") or "").strip().casefold()
    latest_query = str((latest.get("parameters") or {}).get("query") or "").strip().casefold()
    if previous_query and previous_query == latest_query:
        return True
    previous_ids = set(product_ids(str(previous.get("observation") or "")))
    latest_ids = set(product_ids(str(latest.get("observation") or "")))
    return bool(previous_ids) and previous_ids == latest_ids


def build_state_view(
    *,
    task: Mapping,
    trajectory: Mapping,
    latest_observation: str,
    max_steps: int,
) -> ExperienceStateView:
    steps = [step for step in trajectory.get("steps") or [] if isinstance(step, Mapping)]
    query = public_query({**dict(trajectory), "task_id": task["task_id"]})
    tags = constraint_tags(query)
    page_type = observation_page_type(latest_observation)
    current_match = _ASIN_LINE.search(latest_observation)
    current_product_id = current_match.group(1) if current_match else None
    candidate_ids = product_ids(latest_observation)
    guard_reason = _last_guard_reason(trajectory, len(steps))
    selected = _selected_options(steps)
    predicates = set()
    if "available_options:" in latest_observation:
        match = re.search(r"(?m)^available_options:\s*(\{[^\n]*\})", latest_observation)
        try:
            available = json.loads(match.group(1)) if match else {}
        except json.JSONDecodeError:
            available = {}
        if isinstance(available, dict) and available:
            predicates.add("has_multiple_options")
    if "budget" in tags and page_type == "product_detail":
        predicates.add("final_price_unverified")
    if _search_stagnated(steps):
        predicates.update({"search_no_new_candidates", "repeated_search"})
    if steps and steps[-1].get("tool_name") == "open_product":
        predicates.add("candidate_new")
    if guard_reason:
        predicates.add("guard_rejected")
    remaining = max(0, int(max_steps) - len(steps))
    if remaining <= 5:
        predicates.add("remaining_steps_low")
    if any(button.casefold() == "buy now" for button in clickable_buttons(latest_observation)):
        predicates.add("finish_eligible")
    candidate_hash = hashlib.sha256(
        json.dumps(candidate_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ExperienceStateView(
        task_id=int(task["task_id"]),
        public_query=query,
        constraint_tags=tags,
        phase=_phase(steps, guard_reason),
        page_type=page_type,
        current_product_id=current_product_id,
        seen_product_ids=_seen_product_ids(steps, latest_observation),
        candidate_set_hash=candidate_hash,
        unverified_constraints=tags,
        selected_options=selected,
        last_tool_name=(str(steps[-1].get("tool_name") or "") if steps else None),
        last_guard_reason=guard_reason,
        repeat_action_count=_repeat_action_count(steps),
        executed_steps=len(steps),
        remaining_steps=remaining,
        predicates=tuple(sorted(predicates)),
    )


def detect_recall_event(state: ExperienceStateView) -> str | None:
    if state.last_guard_reason:
        return "guard_rejection"
    if state.executed_steps == 0:
        return "task_start"
    if state.last_tool_name == "open_product":
        return "candidate_opened"
    if (
        state.page_type == "product_detail"
        and "finish_eligible" in state.predicates
        and (
            state.last_tool_name == "select_option"
            or state.phase == "detail_verification"
        )
    ):
        return "pre_purchase"
    if "search_no_new_candidates" in state.predicates:
        return "search_stagnation"
    if "remaining_steps_low" in state.predicates and state.page_type != "product_detail":
        return "pre_finish"
    return None
