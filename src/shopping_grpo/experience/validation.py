"""Deterministic paired gates for candidate experiences and store regression."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import random

from shopping_grpo.evaluation.metrics import compute_deterministic_metrics
from shopping_grpo.evaluation.trajectory import normalize_trajectory


VALIDATION_THRESHOLDS_VERSION = "shopping-experience-validation-thresholds-v1"
VALIDATION_RESULT_VERSION = "shopping-experience-validation-result-v1"
_CANDIDATE_THRESHOLD_FIELDS = {
    "minimum_targeted_pairs",
    "strict_success_delta_exclusive_min",
    "strict_success_ci_lower_min",
    "wrong_purchase_increase_max",
    "reward_unverifiable_increase_max",
    "guard_rejection_mean_increase_max",
    "repeat_loop_increase_max",
    "max_steps_increase_max",
    "injection_tokens_per_turn_max",
    "selected_experiences_per_turn_max",
}
_STORE_THRESHOLD_FIELDS = {
    "minimum_pairs",
    "strict_success_delta_min",
    "wrong_purchase_increase_max",
    "reward_unverifiable_increase_max",
    "guard_rejection_mean_increase_max",
    "repeat_loop_increase_max",
    "max_steps_increase_max",
    "infrastructure_invalid_increase_max",
}


def trajectory_key(trajectory: Mapping) -> tuple[int, int]:
    pair_seed = trajectory.get("sampling_seed")
    if pair_seed is None:
        pair_seed = trajectory.get("attempt_index", 0)
    return int(trajectory["task_id"]), int(pair_seed)


def index_trajectories(trajectories: Iterable[Mapping]) -> dict[tuple[int, int], dict]:
    result = {}
    for raw in trajectories:
        trajectory = deepcopy(dict(raw))
        key = trajectory_key(trajectory)
        if key in result:
            raise ValueError(f"duplicate trajectory pair key: {key}")
        result[key] = trajectory
    return result


def _experience_events(trajectory: Mapping) -> list[Mapping]:
    experience = trajectory.get("experience")
    if not isinstance(experience, Mapping):
        return []
    events = experience.get("events")
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, Mapping)]


def selected_experience_ids(trajectory: Mapping) -> set[str]:
    return {
        str(experience_id)
        for event in _experience_events(trajectory)
        for experience_id in event.get("selected_experience_ids") or []
    }


def _facts(trajectory: Mapping) -> dict:
    metrics = compute_deterministic_metrics(normalize_trajectory(trajectory))
    reward = metrics["reward_and_outcome"]
    legality = metrics["legality"]
    repetition = metrics["repetition"]
    validity = metrics["validity"]
    events = _experience_events(trajectory)
    return {
        "strict_success": int(reward["strict_gold_success"]),
        "wrong_purchase": int(reward["reward_type"] == "wrong_purchase"),
        "reward_unverifiable": int(
            reward["reward_type"] == "reward_unverifiable"
        ),
        "guard_rejections": int(legality["guard_rejection_count"]),
        "repeat_loop": int(repetition["environment_repeat_loop"]),
        "max_steps": int(reward["max_steps_termination"]),
        "infrastructure_invalid": int(validity["infrastructure_invalid"]),
        "experience_tokens": sum(int(event.get("experience_tokens", 0)) for event in events),
        "selected_experiences": sum(
            len(event.get("selected_experience_ids") or []) for event in events
        ),
        "experience_turns": len(events),
    }


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    index = round((len(sorted_values) - 1) * float(probability))
    return float(sorted_values[index])


def _bootstrap_success_delta(
    deltas: list[int],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if int(samples) < 1:
        raise ValueError("bootstrap samples must be positive")
    generator = random.Random(int(seed))
    size = len(deltas)
    values = []
    for _ in range(int(samples)):
        values.append(
            sum(deltas[generator.randrange(size)] for _ in range(size)) / size
        )
    values.sort()
    return _quantile(values, 0.025), _quantile(values, 0.975)


def paired_comparison(
    baseline: Mapping[tuple[int, int], Mapping],
    treatment: Mapping[tuple[int, int], Mapping],
    *,
    keys: Iterable[tuple[int, int]] | None = None,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 42,
) -> dict:
    baseline_keys = set(baseline)
    treatment_keys = set(treatment)
    if baseline_keys != treatment_keys:
        raise ValueError("baseline and treatment trajectory keys differ")
    selected_keys = sorted(set(keys) if keys is not None else baseline_keys)
    if not selected_keys or not set(selected_keys) <= baseline_keys:
        raise ValueError("paired comparison keys are empty or unavailable")
    baseline_facts = [_facts(baseline[key]) for key in selected_keys]
    treatment_facts = [_facts(treatment[key]) for key in selected_keys]
    success_deltas = [
        right["strict_success"] - left["strict_success"]
        for left, right in zip(baseline_facts, treatment_facts, strict=True)
    ]
    ci_low, ci_high = _bootstrap_success_delta(
        success_deltas,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )

    def total_delta(field: str) -> int:
        return sum(row[field] for row in treatment_facts) - sum(
            row[field] for row in baseline_facts
        )

    pair_count = len(selected_keys)
    treatment_turns = sum(row["experience_turns"] for row in treatment_facts)
    return {
        "pairs": pair_count,
        "pair_keys": [list(key) for key in selected_keys],
        "strict_success": {
            "baseline_rate": sum(row["strict_success"] for row in baseline_facts)
            / pair_count,
            "treatment_rate": sum(row["strict_success"] for row in treatment_facts)
            / pair_count,
            "paired_delta_mean": sum(success_deltas) / pair_count,
            "bootstrap_ci95": [ci_low, ci_high],
            "wins": sum(delta > 0 for delta in success_deltas),
            "losses": sum(delta < 0 for delta in success_deltas),
            "ties": sum(delta == 0 for delta in success_deltas),
        },
        "count_deltas": {
            field: total_delta(field)
            for field in (
                "wrong_purchase",
                "reward_unverifiable",
                "repeat_loop",
                "max_steps",
                "infrastructure_invalid",
            )
        },
        "guard_rejection_mean_delta": total_delta("guard_rejections") / pair_count,
        "treatment_experience": {
            "mean_tokens_per_pair": sum(
                row["experience_tokens"] for row in treatment_facts
            )
            / pair_count,
            "mean_tokens_per_turn": (
                sum(row["experience_tokens"] for row in treatment_facts)
                / treatment_turns
                if treatment_turns
                else 0.0
            ),
            "mean_selected_per_turn": (
                sum(row["selected_experiences"] for row in treatment_facts)
                / treatment_turns
                if treatment_turns
                else 0.0
            ),
        },
    }


def validate_thresholds(value: object) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("experience validation thresholds must be an object")
    thresholds = deepcopy(dict(value))
    required = {"schema_version", "bootstrap", "candidate", "store"}
    if set(thresholds) != required:
        raise ValueError("experience validation thresholds fields differ from schema")
    if thresholds["schema_version"] != VALIDATION_THRESHOLDS_VERSION:
        raise ValueError("unsupported experience validation threshold version")
    bootstrap = thresholds["bootstrap"]
    if not isinstance(bootstrap, Mapping):
        raise ValueError("bootstrap thresholds must be an object")
    if set(bootstrap) != {"samples", "seed"}:
        raise ValueError("bootstrap thresholds fields differ from schema")
    if int(bootstrap["samples"]) < 1:
        raise ValueError("bootstrap samples must be positive")
    for section in ("candidate", "store"):
        if not isinstance(thresholds[section], Mapping):
            raise ValueError(f"{section} thresholds must be an object")
    if set(thresholds["candidate"]) != _CANDIDATE_THRESHOLD_FIELDS:
        raise ValueError("candidate threshold fields differ from schema")
    if set(thresholds["store"]) != _STORE_THRESHOLD_FIELDS:
        raise ValueError("store threshold fields differ from schema")
    return thresholds


def candidate_gate(comparison: Mapping, thresholds: Mapping) -> tuple[bool, list[str]]:
    checks = {
        "minimum_targeted_pairs": comparison["pairs"]
        >= int(thresholds["minimum_targeted_pairs"]),
        "strict_success_ci_lower": comparison["strict_success"]["bootstrap_ci95"][0]
        >= float(thresholds["strict_success_ci_lower_min"]),
        "strict_success_mean_improvement": comparison["strict_success"][
            "paired_delta_mean"
        ]
        > float(thresholds["strict_success_delta_exclusive_min"]),
        "wrong_purchase": comparison["count_deltas"]["wrong_purchase"]
        <= int(thresholds["wrong_purchase_increase_max"]),
        "reward_unverifiable": comparison["count_deltas"]["reward_unverifiable"]
        <= int(thresholds["reward_unverifiable_increase_max"]),
        "guard_rejections": comparison["guard_rejection_mean_delta"]
        <= float(thresholds["guard_rejection_mean_increase_max"]),
        "repeat_loop": comparison["count_deltas"]["repeat_loop"]
        <= int(thresholds["repeat_loop_increase_max"]),
        "max_steps": comparison["count_deltas"]["max_steps"]
        <= int(thresholds["max_steps_increase_max"]),
        "injection_tokens": comparison["treatment_experience"]["mean_tokens_per_turn"]
        <= float(thresholds["injection_tokens_per_turn_max"]),
        "selected_experiences": comparison["treatment_experience"][
            "mean_selected_per_turn"
        ]
        <= float(thresholds["selected_experiences_per_turn_max"]),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return not failures, failures


def store_gate(comparison: Mapping, thresholds: Mapping) -> tuple[bool, list[str]]:
    checks = {
        "minimum_pairs": comparison["pairs"] >= int(thresholds["minimum_pairs"]),
        "strict_success_non_inferiority": comparison["strict_success"][
            "paired_delta_mean"
        ]
        >= float(thresholds["strict_success_delta_min"]),
        "wrong_purchase": comparison["count_deltas"]["wrong_purchase"]
        <= int(thresholds["wrong_purchase_increase_max"]),
        "reward_unverifiable": comparison["count_deltas"]["reward_unverifiable"]
        <= int(thresholds["reward_unverifiable_increase_max"]),
        "guard_rejections": comparison["guard_rejection_mean_delta"]
        <= float(thresholds["guard_rejection_mean_increase_max"]),
        "repeat_loop": comparison["count_deltas"]["repeat_loop"]
        <= int(thresholds["repeat_loop_increase_max"]),
        "max_steps": comparison["count_deltas"]["max_steps"]
        <= int(thresholds["max_steps_increase_max"]),
        "infrastructure_invalid": comparison["count_deltas"][
            "infrastructure_invalid"
        ]
        <= int(thresholds["infrastructure_invalid_increase_max"]),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return not failures, failures
