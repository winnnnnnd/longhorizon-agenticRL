"""Deterministic, task-disjoint splits for experience discovery and validation."""

from __future__ import annotations

import hashlib


def stable_experience_task_split(
    task_ids,
    *,
    candidate_dev_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Return discovery and candidate-dev task IDs using a stable hash order."""

    normalized = [int(task_id) for task_id in task_ids]
    if len(normalized) != len(set(normalized)):
        raise ValueError("experience task source contains duplicate task IDs")
    if len(normalized) < 2:
        raise ValueError("experience task splitting requires at least two tasks")
    if not 0.0 < float(candidate_dev_ratio) < 1.0:
        raise ValueError("candidate_dev_ratio must be within (0, 1)")
    ordered = sorted(
        normalized,
        key=lambda task_id: (
            hashlib.sha256(f"{int(seed)}:{task_id}".encode("utf-8")).hexdigest(),
            task_id,
        ),
    )
    candidate_count = max(1, round(len(ordered) * float(candidate_dev_ratio)))
    candidate_count = min(candidate_count, len(ordered) - 1)
    candidate_dev = sorted(ordered[:candidate_count])
    discovery = sorted(ordered[candidate_count:])
    return discovery, candidate_dev


def overlap_report(
    *,
    source_ids,
    discovery_ids,
    candidate_dev_ids,
    grpo_validation_ids=(),
    final_evaluation_ids=(),
    sft_ids=(),
) -> dict:
    """Build explicit overlap counts without silently repairing a bad split."""

    source = set(map(int, source_ids))
    discovery = set(map(int, discovery_ids))
    candidate_dev = set(map(int, candidate_dev_ids))
    grpo_validation = set(map(int, grpo_validation_ids))
    final_evaluation = set(map(int, final_evaluation_ids))
    sft = set(map(int, sft_ids))
    return {
        "source_vs_grpo_validation": sorted(source & grpo_validation),
        "source_vs_final_evaluation": sorted(source & final_evaluation),
        "source_vs_sft": sorted(source & sft),
        "discovery_vs_candidate_dev": sorted(discovery & candidate_dev),
        "discovery_vs_final_evaluation": sorted(discovery & final_evaluation),
        "candidate_dev_vs_final_evaluation": sorted(candidate_dev & final_evaluation),
    }
