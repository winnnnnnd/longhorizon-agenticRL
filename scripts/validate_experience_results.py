#!/usr/bin/env python3
"""Apply frozen paired gates and emit promotion decisions for candidate experiences."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from shopping_grpo.evaluation.artifacts import (
    iter_jsonl,
    load_json,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.evaluation.manifest import sha256_file
from shopping_grpo.experience.contracts import validate_experience_card
from shopping_grpo.experience.promotion import PROMOTION_DECISION_VERSION
from shopping_grpo.experience.validation import (
    VALIDATION_RESULT_VERSION,
    candidate_gate,
    index_trajectories,
    paired_comparison,
    selected_experience_ids,
    store_gate,
    validate_thresholds,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-baseline", type=Path, required=True)
    parser.add_argument("--candidate-treatment", type=Path, required=True)
    parser.add_argument("--store-baseline", type=Path, required=True)
    parser.add_argument("--store-treatment", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("configs/experience_validation.json"),
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument(
        "--forbidden-tasks",
        type=Path,
        action="append",
        default=[Path("data/evaluation/tasks.jsonl")],
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _assert_iso_time(value: str) -> None:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit("--created-at must be ISO-8601") from exc


def _load_index(path: Path):
    return index_trajectories(iter_jsonl(path))


def _assert_no_forbidden_tasks(indexes, forbidden_paths):
    forbidden = set()
    sources = []
    for path in forbidden_paths:
        rows = list(iter_jsonl(path))
        forbidden.update(int(row["task_id"]) for row in rows)
        sources.append(
            {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}
        )
    used = {task_id for index in indexes for task_id, _ in index}
    overlap = sorted(used & forbidden)
    if overlap:
        raise SystemExit(f"validation inputs contain forbidden task IDs: {overlap}")
    return sources


def _targeted_keys(treatment, experience_id: str):
    return sorted(
        key
        for key, trajectory in treatment.items()
        if experience_id in selected_experience_ids(trajectory)
    )


def main():
    args = parse_args()
    _assert_iso_time(args.created_at)
    thresholds = validate_thresholds(load_json(args.thresholds))
    bootstrap = thresholds["bootstrap"]
    cards = [validate_experience_card(card) for card in iter_jsonl(args.candidates)]
    if any(card["status"] not in {"candidate", "validated"} for card in cards):
        raise SystemExit("validation input cards must be candidate or validated")
    card_keys = [(card["experience_id"], card["revision"]) for card in cards]
    if len(card_keys) != len(set(card_keys)):
        raise SystemExit("candidate file contains duplicate experience ID/revision")

    candidate_baseline = _load_index(args.candidate_baseline)
    candidate_treatment = _load_index(args.candidate_treatment)
    store_baseline = _load_index(args.store_baseline)
    store_treatment = _load_index(args.store_treatment)
    forbidden_sources = _assert_no_forbidden_tasks(
        (candidate_baseline, candidate_treatment, store_baseline, store_treatment),
        args.forbidden_tasks,
    )
    if set(candidate_baseline) != set(candidate_treatment):
        raise SystemExit("candidate baseline/treatment pair keys differ")
    store_comparison = paired_comparison(
        store_baseline,
        store_treatment,
        bootstrap_samples=int(bootstrap["samples"]),
        bootstrap_seed=int(bootstrap["seed"]),
    )
    store_passed, store_failures = store_gate(
        store_comparison,
        thresholds["store"],
    )

    results = []
    gate_by_card = {}
    ordered_cards = sorted(
        cards,
        key=lambda row: (row["experience_id"], row["revision"]),
    )
    for offset, card in enumerate(ordered_cards):
        experience_id = card["experience_id"]
        targeted = _targeted_keys(candidate_treatment, experience_id)
        if targeted:
            comparison = paired_comparison(
                candidate_baseline,
                candidate_treatment,
                keys=targeted,
                bootstrap_samples=int(bootstrap["samples"]),
                bootstrap_seed=int(bootstrap["seed"]) + offset + 1,
            )
            candidate_passed, candidate_failures = candidate_gate(
                comparison,
                thresholds["candidate"],
            )
        else:
            comparison = None
            candidate_passed = False
            candidate_failures = ["no_targeted_pairs"]
        key = (experience_id, card["revision"])
        gate_by_card[key] = (candidate_passed, candidate_failures)
        results.append(
            {
                "schema_version": VALIDATION_RESULT_VERSION,
                "experience_id": experience_id,
                "revision": card["revision"],
                "candidate_validation": {
                    "passed": candidate_passed,
                    "failures": candidate_failures,
                    "comparison": comparison,
                },
                "store_regression": {
                    "passed": store_passed,
                    "failures": store_failures,
                    "comparison": store_comparison,
                },
            }
        )
    write_jsonl_atomic(args.results, results, force=args.force)

    input_paths = {
        "candidates": args.candidates,
        "candidate_baseline": args.candidate_baseline,
        "candidate_treatment": args.candidate_treatment,
        "store_baseline": args.store_baseline,
        "store_treatment": args.store_treatment,
        "thresholds": args.thresholds,
    }
    manifest = {
        "schema_version": "shopping-experience-validation-manifest-v1",
        "created_at": args.created_at,
        "thresholds": thresholds,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in input_paths.items()
        },
        "forbidden_task_sources": forbidden_sources,
        "forbidden_task_overlap": [],
        "results": {
            "path": str(args.results),
            "rows": len(results),
            "sha256": sha256_file(args.results),
        },
    }
    write_json_atomic(args.manifest, manifest, force=args.force)
    validation_manifest_hash = sha256_file(args.manifest)

    decisions = []
    for card in sorted(cards, key=lambda row: (row["experience_id"], row["revision"])):
        key = (card["experience_id"], card["revision"])
        candidate_passed, candidate_failures = gate_by_card[key]
        promoted = candidate_passed and store_passed
        reasons = [f"candidate:{failure}" for failure in candidate_failures]
        reasons.extend(f"store:{failure}" for failure in store_failures)
        decisions.append(
            {
                "schema_version": PROMOTION_DECISION_VERSION,
                "experience_id": card["experience_id"],
                "revision": card["revision"],
                "candidate_validation_passed": candidate_passed,
                "store_regression_passed": store_passed,
                "validation_manifest_hash": validation_manifest_hash,
                "decision": "promote" if promoted else "reject",
                "reason": "; ".join(reasons) if reasons else "all frozen gates passed",
            }
        )
    write_jsonl_atomic(args.decisions, decisions, force=args.force)


if __name__ == "__main__":
    main()
