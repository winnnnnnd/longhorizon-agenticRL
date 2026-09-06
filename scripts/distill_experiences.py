#!/usr/bin/env python3
"""Jointly distill Teacher and repeated Student rollouts into candidate experiences."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from shopping_grpo.evaluation.artifacts import iter_jsonl, write_json_atomic, write_jsonl_atomic
from shopping_grpo.evaluation.manifest import sha256_file
from shopping_grpo.experience.config import load_experience_config
from shopping_grpo.experience.distillation import (
    DISTILLATION_PROMPT_VERSION,
    MultiTrajectoryDistiller,
    build_joint_analysis_groups,
    build_trajectory_records,
    merge_duplicate_cards,
)
from shopping_grpo.experience.factory import build_distillation_client


def parse_args():
    parser = argparse.ArgumentParser(
        description="从多条 Teacher/Student trajectory 联合沉淀候选经验"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experience.json"))
    parser.add_argument(
        "--teacher",
        type=Path,
        action="append",
        default=[],
        help="Curated Teacher gold transcript JSONL; may omit terminal Reward fields.",
    )
    parser.add_argument(
        "--teacher-raw",
        type=Path,
        action="append",
        default=[],
        help="Raw Teacher rollout JSONL with real terminal outcomes.",
    )
    parser.add_argument("--student", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--created-at",
        required=True,
        help="固定 ISO-8601 时间；重建相同候选快照时必须复用同一值。",
    )
    parser.add_argument(
        "--forbidden-tasks",
        type=Path,
        action="append",
        default=[Path("data/evaluation/tasks.jsonl")],
        help="这些 task ID 不得进入经验抽取；默认包含 Final-200。",
    )
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.teacher and not args.teacher_raw and not args.student:
        raise SystemExit(
            "至少提供一个 --teacher、--teacher-raw 或 --student trajectory 文件"
        )
    try:
        datetime.fromisoformat(args.created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit("--created-at must be ISO-8601") from exc
    config = load_experience_config(args.config)
    records = []
    source_manifest = []
    for path in args.teacher:
        rows = list(iter_jsonl(path))
        records.extend(
            build_trajectory_records(
                rows,
                actor_role="teacher",
                source_kind="curated_teacher_gold",
            )
        )
        source_manifest.append(
            {
                "role": "teacher",
                "source_kind": "curated_teacher_gold",
                "path": str(path),
                "rows": len(rows),
                "sha256": sha256_file(path),
            }
        )
    for path in args.teacher_raw:
        rows = list(iter_jsonl(path))
        records.extend(
            build_trajectory_records(
                rows,
                actor_role="teacher",
                source_kind="teacher_raw_rollout",
            )
        )
        source_manifest.append(
            {
                "role": "teacher",
                "source_kind": "teacher_raw_rollout",
                "path": str(path),
                "rows": len(rows),
                "sha256": sha256_file(path),
            }
        )
    for path in args.student:
        rows = list(iter_jsonl(path))
        records.extend(
            build_trajectory_records(
                rows,
                actor_role="student",
                source_kind="agent_rollout",
            )
        )
        source_manifest.append(
            {
                "role": "student",
                "source_kind": "agent_rollout",
                "path": str(path),
                "rows": len(rows),
                "sha256": sha256_file(path),
            }
        )
    forbidden_sources = []
    forbidden_task_ids = set()
    for path in args.forbidden_tasks:
        rows = list(iter_jsonl(path))
        task_ids = {int(row["task_id"]) for row in rows}
        forbidden_task_ids.update(task_ids)
        forbidden_sources.append(
            {
                "path": str(path),
                "rows": len(rows),
                "sha256": sha256_file(path),
            }
        )
    leaked = sorted(
        {record["task_id"] for record in records} & forbidden_task_ids
    )
    if leaked:
        raise SystemExit(f"trajectory sources contain forbidden task IDs: {leaked}")
    distillation_config = config["distillation"]
    groups = build_joint_analysis_groups(
        records,
        maximum_group_size=int(distillation_config.get("maximum_group_size", 8)),
        similarity_threshold=float(distillation_config.get("similarity_threshold", 0.25)),
    )
    if args.max_groups > 0:
        groups = groups[: args.max_groups]
    distiller = MultiTrajectoryDistiller(
        build_distillation_client(args.config),
        created_at=args.created_at,
    )
    cards = []
    audits = []
    for group in groups:
        group_cards, audit = distiller.distill_group(group)
        cards.extend(group_cards)
        audits.append(audit)
    cards = merge_duplicate_cards(cards)
    write_jsonl_atomic(args.output, cards, force=args.force)
    write_jsonl_atomic(args.audit_output, audits, force=args.force)
    manifest = {
        "schema_version": "shopping-experience-distillation-manifest-v1",
        "created_at": args.created_at,
        "prompt_version": DISTILLATION_PROMPT_VERSION,
        "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
        "sources": source_manifest,
        "forbidden_task_sources": forbidden_sources,
        "forbidden_task_overlap": [],
        "trajectory_records": len(records),
        "analysis_groups": len(groups),
        "candidate_experiences": len(cards),
        "outputs": {
            "cards": {"path": str(args.output), "sha256": sha256_file(args.output)},
            "audit": {"path": str(args.audit_output), "sha256": sha256_file(args.audit_output)},
        },
    }
    write_json_atomic(args.manifest, manifest, force=args.force)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
