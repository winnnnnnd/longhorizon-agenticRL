"""Convert raw or sanitized shopping trajectories into actor-visible decision windows."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import re

from shopping_grpo.evaluation.trajectory import (
    NORMALIZED_TRAJECTORY_VERSION,
    normalize_trajectory,
)
from shopping_grpo.environment.actions import product_ids
from shopping_grpo.experience.contracts import SOURCE_KINDS


WINDOW_VERSION = "shopping-experience-decision-window-v1"
SEGMENT_VERSION = "shopping-experience-segment-v1"
_INSTRUCTION_PREFIX = re.compile(r"^\s*Instruction:\s*", re.I)
_ACTOR_VISIBLE_EVENT_FIELDS = (
    "event_id",
    "event_type",
    "assistant_text",
    "tool_name",
    "parameters",
    "tool_call_parse_error",
    "guard_reason",
    "actor_visible_observation",
)


def public_query(trajectory: Mapping) -> str:
    initial = trajectory.get("initial_result")
    if isinstance(initial, Mapping):
        instruction = initial.get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            return _INSTRUCTION_PREFIX.sub("", instruction).strip()
    for message in trajectory.get("messages") or []:
        if isinstance(message, Mapping) and message.get("role") == "user":
            return _INSTRUCTION_PREFIX.sub("", str(message.get("content") or "")).strip()
    return _INSTRUCTION_PREFIX.sub(
        "", str(trajectory.get("actor_query") or "")
    ).strip()


def _arguments(tool_call: Mapping) -> dict:
    function = tool_call.get("function")
    function = function if isinstance(function, Mapping) else {}
    raw = function.get("arguments") or {}
    if isinstance(raw, Mapping):
        return deepcopy(dict(raw))
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return deepcopy(value) if isinstance(value, dict) else {}


def observation_page_type(observation: object) -> str:
    text = str(observation or "")
    match = re.search(r"(?m)^page_type:\s*([^\n]+)", text)
    if match:
        return match.group(1).strip()
    if re.search(r"(?m)^\d+\|\d{8,12}\|", text):
        return "search_results"
    if re.search(r"(?m)^asin:\s*\d{8,12}\s*$", text):
        return "product_detail"
    return "unknown"


def decision_phase(event: Mapping) -> str:
    if event.get("event_type") == "guard_rejection" or event.get("guard_reason"):
        return "error_recovery"
    tool_name = str(event.get("tool_name") or "")
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
    }.get(tool_name, "task_understanding")


def _events_from_messages(messages: object) -> list[dict]:
    if not isinstance(messages, list):
        return []
    tool_messages = {
        str(message.get("tool_call_id")): message
        for message in messages
        if isinstance(message, Mapping)
        and message.get("role") == "tool"
        and message.get("tool_call_id") is not None
    }
    events = []
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, Mapping):
                continue
            function = tool_call.get("function")
            function = function if isinstance(function, Mapping) else {}
            call_id = str(tool_call.get("id") or "")
            tool_message = tool_messages.get(call_id, {})
            observation = str(tool_message.get("content") or "")
            event = {
                "event_id": f"e{len(events) + 1:04d}",
                "event_type": (
                    "guard_rejection"
                    if tool_message.get("runtime_action_guard") is True
                    else "tool_step"
                ),
                "assistant_text": str(message.get("content") or ""),
                "tool_call_id": call_id or None,
                "tool_name": str(function.get("name") or tool_message.get("name") or ""),
                "parameters": _arguments(tool_call),
                "actor_visible_observation": observation,
                "page_type": observation_page_type(observation),
                "guard_reason": None,
                "done": "page_type: terminal" in observation,
            }
            if event["event_type"] == "guard_rejection":
                match = re.search(r"守卫拒绝（([^）]+)）", observation)
                event["guard_reason"] = match.group(1) if match else "unknown"
            event["phase"] = decision_phase(event)
            events.append(event)
    return events


def actor_visible_events(trajectory: Mapping) -> list[dict]:
    if trajectory.get("schema_version") == NORMALIZED_TRAJECTORY_VERSION:
        events = deepcopy(trajectory.get("events") or [])
    elif trajectory.get("steps") is not None:
        events = normalize_trajectory(trajectory).get("events") or []
    else:
        return _events_from_messages(trajectory.get("messages"))
    result = []
    for index, raw_event in enumerate(events, start=1):
        if not isinstance(raw_event, Mapping):
            continue
        event = {
            field: deepcopy(raw_event.get(field))
            for field in _ACTOR_VISIBLE_EVENT_FIELDS
            if field in raw_event
        }
        event["event_id"] = str(event.get("event_id") or f"e{index:04d}")
        observation = str(event.get("actor_visible_observation") or "")
        event["page_type"] = observation_page_type(observation)
        event["phase"] = decision_phase(event)
        result.append(event)
    return result


def _segment_state_predicates(events: list[Mapping]) -> list[str]:
    """Derive only predicates that are directly observable inside a segment."""

    predicates = set()
    searches = []
    search_result_sets = []
    for event in events:
        tool_name = str(event.get("tool_name") or "")
        parameters = event.get("parameters") or {}
        observation = str(event.get("actor_visible_observation") or "")
        if event.get("event_type") == "guard_rejection" or event.get("guard_reason"):
            predicates.add("guard_rejected")
        if tool_name == "open_product":
            predicates.add("candidate_new")
        if "available_options:" in observation:
            predicates.add("has_multiple_options")
        if "buy now" in observation.casefold():
            predicates.add("finish_eligible")
        if tool_name == "search_products":
            query = str(parameters.get("query") or "").strip().casefold()
            searches.append(query)
            search_result_sets.append(tuple(product_ids(observation)))

    nonempty_searches = [query for query in searches if query]
    if len(nonempty_searches) != len(set(nonempty_searches)):
        predicates.add("repeated_search")
    if any(
        left and left == right
        for left, right in zip(search_result_sets, search_result_sets[1:])
    ):
        predicates.add("search_no_new_candidates")
    return sorted(predicates)


def build_trajectory_segments(
    trajectory: Mapping,
    *,
    source_kind: str,
    actor_role: str,
    context_events: int = 2,
) -> list[dict]:
    """Split a trajectory into contiguous decision-phase segments.

    A segment owns consecutive events with the same deterministic phase.  Small
    read-only context windows are attached on both sides, while the segment's
    own events remain explicit.  Sequential LLM extraction can therefore use
    earlier derived knowledge without flattening the whole trajectory.
    """

    if int(context_events) < 0:
        raise ValueError("context_events cannot be negative")
    if actor_role not in {"teacher", "student"}:
        raise ValueError("actor_role must be teacher or student")
    if source_kind not in SOURCE_KINDS:
        raise ValueError("source_kind is unsupported")
    events = actor_visible_events(trajectory)
    if not events:
        return []
    outcome = trajectory_outcome(trajectory, source_kind)
    trajectory_id = str(trajectory.get("trajectory_id") or "")
    if not trajectory_id:
        raise ValueError("trajectory segment requires trajectory_id")
    task_id = int(trajectory["task_id"])
    query = public_query(trajectory)
    if not query:
        raise ValueError("trajectory segment requires a public query")

    spans = []
    start = 0
    for index in range(1, len(events) + 1):
        if index == len(events) or events[index]["phase"] != events[start]["phase"]:
            spans.append((start, index))
            start = index

    segments = []
    context_size = int(context_events)
    for segment_index, (start, stop) in enumerate(spans, start=1):
        owned_events = deepcopy(events[start:stop])
        segment_id = f"{trajectory_id}:seg{segment_index:04d}"
        segments.append(
            {
                "schema_version": SEGMENT_VERSION,
                "segment_id": segment_id,
                "segment_index": segment_index - 1,
                "trajectory_id": trajectory_id,
                "task_id": task_id,
                "actor_role": actor_role,
                "source_kind": str(source_kind),
                "public_query": query,
                "phase": owned_events[0]["phase"],
                "event_range": {
                    "start_event_id": owned_events[0]["event_id"],
                    "end_event_id": owned_events[-1]["event_id"],
                },
                "context_before": deepcopy(events[max(0, start - context_size) : start]),
                "events": owned_events,
                "context_after": deepcopy(events[stop : stop + context_size]),
                "observed_state_predicates": _segment_state_predicates(owned_events),
                "outcome": deepcopy(outcome),
            }
        )
    return segments


def trajectory_outcome(trajectory: Mapping, source_kind: str) -> dict:
    if source_kind == "curated_teacher_gold":
        return {
            "status": "done",
            "reward_type": "gold_purchase",
            "reward_valid": True,
            "strict_success": True,
            "evidence_level": "dataset_manifest",
        }
    terminal = trajectory.get("terminal_result")
    terminal = terminal if isinstance(terminal, Mapping) else trajectory.get("terminal")
    terminal = terminal if isinstance(terminal, Mapping) else {}
    reward = terminal.get("reward_detail")
    reward = reward if isinstance(reward, Mapping) else {}
    reward_type = str(
        reward.get("reward_type")
        or terminal.get("termination_reason")
        or trajectory.get("termination_reason")
        or trajectory.get("status")
        or "unknown"
    )
    reward_valid = reward.get("reward_valid")
    return {
        "status": str(trajectory.get("status") or "unknown"),
        "reward_type": reward_type,
        "reward_valid": reward_valid,
        "strict_success": reward_type == "gold_purchase" and reward_valid is True,
        "evidence_level": "raw_terminal_result",
    }


def build_decision_windows(
    trajectory: Mapping,
    *,
    source_kind: str,
    actor_role: str,
    before: int = 2,
    after: int = 2,
) -> list[dict]:
    """Build bounded local windows while keeping only actor-visible event content."""

    events = actor_visible_events(trajectory)
    outcome = trajectory_outcome(trajectory, source_kind)
    trajectory_id = str(trajectory.get("trajectory_id") or "")
    task_id = int(trajectory["task_id"])
    query = public_query(trajectory)
    windows = []
    for index, event in enumerate(events):
        start = max(0, index - int(before))
        stop = min(len(events), index + int(after) + 1)
        windows.append(
            {
                "schema_version": WINDOW_VERSION,
                "trajectory_id": trajectory_id,
                "task_id": task_id,
                "actor_role": str(actor_role),
                "source_kind": str(source_kind),
                "public_query": query,
                "phase": event["phase"],
                "focal_event_id": event["event_id"],
                "events": deepcopy(events[start:stop]),
                "outcome": deepcopy(outcome),
            }
        )
    return windows
