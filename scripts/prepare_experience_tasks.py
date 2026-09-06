#!/usr/bin/env python3
"""Create deterministic discovery/dev task lists for the external-experience route."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from shopping_grpo.evaluation.artifacts import iter_jsonl, write_json_atomic, write_jsonl_atomic
from shopping_grpo.evaluation.manifest import sha256_file
from shopping_grpo.experience.contracts import canonical_sha256
from shopping_grpo.experience.task_split import overlap_report, stable_experience_task_split


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/grpo/train.jsonl"))
    parser.add_argument(
        "--grpo-validation",
        type=Path,
        default=Path("data/grpo/validation.jsonl"),
    )
    parser.add_argument(
        "--final-evaluation",
        type=Path,
        default=Path("data/evaluation/tasks.jsonl"),
    )
    parser.add_argument("--sft", type=Path, default=Path("data/sft_pure_v4/all.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experience/tasks"))
    parser.add_argument("--candidate-dev-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _task_ids(path: Path) -> list[int]:
    ids = []
    for row_number, row in enumerate(iter_jsonl(path), start=1):
        task_id = row.get("task_id")
        if not isinstance(task_id, int) or isinstance(task_id, bool):
            raise ValueError(f"{path}:{row_number}: task_id must be an integer")
        ids.append(task_id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate task IDs")
    return ids


def main():
    args = parse_args()
    source_ids = _task_ids(args.source)
    validation_ids = _task_ids(args.grpo_validation)
    final_ids = _task_ids(args.final_evaluation)
    sft_ids = _task_ids(args.sft)
    discovery, candidate_dev = stable_experience_task_split(
        source_ids,
        candidate_dev_ratio=args.candidate_dev_ratio,
        seed=args.seed,
    )
    overlaps = overlap_report(
        source_ids=source_ids,
        discovery_ids=discovery,
        candidate_dev_ids=candidate_dev,
        grpo_validation_ids=validation_ids,
        final_evaluation_ids=final_ids,
        sft_ids=sft_ids,
    )
    forbidden = {
        key: value
        for key, value in overlaps.items()
        if key != "source_vs_sft" and value
    }
    if forbidden:
        raise ValueError(f"experience task split has forbidden overlap: {forbidden}")

    discovery_path = args.output_dir / "experience_discovery.jsonl"
    candidate_dev_path = args.output_dir / "experience_candidate_dev.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    write_jsonl_atomic(
        discovery_path,
        ({"task_id": task_id} for task_id in discovery),
        force=args.force,
    )
    write_jsonl_atomic(
        candidate_dev_path,
        ({"task_id": task_id} for task_id in candidate_dev),
        force=args.force,
    )
    manifest = {
        "schema_version": "shopping-experience-task-split-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "candidate_dev_ratio": args.candidate_dev_ratio,
        "sources": {
            "grpo_train": {"path": str(args.source), "sha256": sha256_file(args.source)},
            "grpo_validation": {
                "path": str(args.grpo_validation),
                "sha256": sha256_file(args.grpo_validation),
            },
            "final_evaluation": {
                "path": str(args.final_evaluation),
                "sha256": sha256_file(args.final_evaluation),
            },
            "sft": {"path": str(args.sft), "sha256": sha256_file(args.sft)},
        },
        "splits": {
            "experience_discovery": {
                "path": str(discovery_path),
                "tasks": len(discovery),
                "task_ids_hash": canonical_sha256(discovery),
                "sha256": sha256_file(discovery_path),
            },
            "experience_candidate_dev": {
                "path": str(candidate_dev_path),
                "tasks": len(candidate_dev),
                "task_ids_hash": canonical_sha256(candidate_dev),
                "sha256": sha256_file(candidate_dev_path),
            },
        },
        "overlap_checks": overlaps,
    }
    write_json_atomic(manifest_path, manifest, force=args.force)


if __name__ == "__main__":
    main()
