#!/usr/bin/env python3
"""Extract trajectory segments, cluster their keys and distill Experience Cards."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from shopping_grpo.evaluation.artifacts import (
    iter_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.evaluation.manifest import sha256_file
from shopping_grpo.experience.config import load_experience_config
from shopping_grpo.experience.distillation import merge_duplicate_cards
from shopping_grpo.experience.factory import build_distillation_client
from shopping_grpo.experience.segment_extraction import (
    CLUSTER_DISTILLATION_PROMPT_VERSION,
    SEGMENT_EXTRACTION_PROMPT_VERSION,
    SegmentClusterDistiller,
    SequentialSegmentExtractor,
    apply_latest_card_revisions,
    cluster_segment_extractions,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "按阶段切分 Teacher/Student trajectory，顺序抽取结构化 Segment，"
            "按六维键聚类并合成候选 Experience Card"
        )
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
    parser.add_argument(
        "--segments-output",
        type=Path,
        help="结构化 Segment JSONL；默认与 --output 同目录。",
    )
    parser.add_argument(
        "--clusters-output",
        type=Path,
        help="六维聚类 JSONL；默认与 --output 同目录。",
    )
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--previous-cards",
        type=Path,
        help="上一版本 Card JSONL；同一 experience_id 由新 revision 覆盖。",
    )
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
    parser.add_argument("--max-clusters", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _default_side_output(cards_path: Path, suffix: str) -> Path:
    return cards_path.with_name(f"{cards_path.stem}.{suffix}.jsonl")


def _load_sources(args) -> tuple[list[dict], list[dict]]:
    inputs = []
    source_manifest = []
    source_specs = (
        (args.teacher, "teacher", "curated_teacher_gold"),
        (args.teacher_raw, "teacher", "teacher_raw_rollout"),
        (args.student, "student", "agent_rollout"),
    )
    for paths, actor_role, source_kind in source_specs:
        for path in paths:
            rows = list(iter_jsonl(path))
            inputs.extend(
                {
                    "trajectory": row,
                    "actor_role": actor_role,
                    "source_kind": source_kind,
                }
                for row in rows
            )
            source_manifest.append(
                {
                    "role": actor_role,
                    "source_kind": source_kind,
                    "path": str(path),
                    "rows": len(rows),
                    "sha256": sha256_file(path),
                }
            )
    return inputs, source_manifest


def _forbidden_tasks(paths: list[Path]) -> tuple[set[int], list[dict]]:
    forbidden_task_ids = set()
    sources = []
    for path in paths:
        rows = list(iter_jsonl(path))
        forbidden_task_ids.update(int(row["task_id"]) for row in rows)
        sources.append(
            {
                "path": str(path),
                "rows": len(rows),
                "sha256": sha256_file(path),
            }
        )
    return forbidden_task_ids, sources


def _assert_distinct_paths(args, segments_output: Path, clusters_output: Path) -> None:
    outputs = {
        "cards": args.output,
        "segments": segments_output,
        "clusters": clusters_output,
        "audit": args.audit_output,
        "manifest": args.manifest,
    }
    resolved_outputs = {name: path.resolve() for name, path in outputs.items()}
    if len(set(resolved_outputs.values())) != len(resolved_outputs):
        raise SystemExit("experience output paths must be distinct")
    input_paths = [
        args.config,
        *args.teacher,
        *args.teacher_raw,
        *args.student,
        *args.forbidden_tasks,
    ]
    if args.previous_cards:
        input_paths.append(args.previous_cards)
    overlap = set(resolved_outputs.values()) & {path.resolve() for path in input_paths}
    if overlap:
        raise SystemExit(
            "experience output path overlaps an input: "
            + ", ".join(sorted(str(path) for path in overlap))
        )


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
    if args.max_clusters < 0:
        raise SystemExit("--max-clusters cannot be negative")

    segments_output = args.segments_output or _default_side_output(
        args.output, "segments"
    )
    clusters_output = args.clusters_output or _default_side_output(
        args.output, "clusters"
    )
    _assert_distinct_paths(args, segments_output, clusters_output)

    config = load_experience_config(args.config)
    inputs, source_manifest = _load_sources(args)
    trajectory_ids = [
        str(item["trajectory"].get("trajectory_id") or "") for item in inputs
    ]
    if any(not trajectory_id for trajectory_id in trajectory_ids):
        raise SystemExit("trajectory source contains a missing trajectory_id")
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise SystemExit("trajectory sources contain duplicate trajectory IDs")
    forbidden_task_ids, forbidden_sources = _forbidden_tasks(args.forbidden_tasks)
    input_task_ids = {int(item["trajectory"]["task_id"]) for item in inputs}
    leaked = sorted(input_task_ids & forbidden_task_ids)
    if leaked:
        raise SystemExit(f"trajectory sources contain forbidden task IDs: {leaked}")

    distillation_config = config["distillation"]
    client = build_distillation_client(args.config)
    extractor = SequentialSegmentExtractor(
        client,
        context_events=int(distillation_config.get("context_events", 2)),
        maximum_prior_knowledge=int(
            distillation_config.get("maximum_prior_knowledge", 24)
        ),
    )
    extractions = []
    audits = []
    segmentless_trajectories = []
    for item in inputs:
        trajectory_extractions, trajectory_audits = extractor.extract_trajectory(
            item["trajectory"],
            actor_role=item["actor_role"],
            source_kind=item["source_kind"],
        )
        extractions.extend(trajectory_extractions)
        if not trajectory_extractions:
            trajectory_id = str(item["trajectory"]["trajectory_id"])
            segmentless_trajectories.append(trajectory_id)
            audits.append(
                {
                    "stage": "trajectory_segmentation",
                    "trajectory_id": trajectory_id,
                    "segment_count": 0,
                    "reason": "no_actor_visible_events",
                }
            )
        audits.extend(
            {"stage": "segment_extraction", **audit}
            for audit in trajectory_audits
        )

    all_clusters = cluster_segment_extractions(extractions)
    selected_clusters = all_clusters
    if args.max_clusters > 0:
        selected_clusters = all_clusters[: args.max_clusters]
    distiller = SegmentClusterDistiller(
        client,
        created_at=args.created_at,
        maximum_cluster_members=int(
            distillation_config.get("maximum_cluster_members", 16)
        ),
    )
    raw_cards = []
    for cluster in selected_clusters:
        card, audit = distiller.distill_cluster(cluster)
        audits.append({"stage": "cluster_distillation", **audit})
        if card is not None:
            raw_cards.append(card)

    previous_cards = (
        list(iter_jsonl(args.previous_cards)) if args.previous_cards else []
    )
    merged_cards = merge_duplicate_cards(raw_cards)
    retained_by_content_hash = {
        card["content_hash"]: card["experience_id"] for card in merged_cards
    }
    for audit in audits:
        if audit["stage"] == "cluster_distillation" and audit["content_hash"]:
            audit["retained_experience_id"] = retained_by_content_hash[
                audit["content_hash"]
            ]
    cards = apply_latest_card_revisions(
        merged_cards,
        previous_cards,
    )
    write_jsonl_atomic(segments_output, extractions, force=args.force)
    write_jsonl_atomic(clusters_output, all_clusters, force=args.force)
    write_jsonl_atomic(args.output, cards, force=args.force)
    write_jsonl_atomic(args.audit_output, audits, force=args.force)

    outputs = {
        "segments": {
            "path": str(segments_output),
            "rows": len(extractions),
            "sha256": sha256_file(segments_output),
        },
        "clusters": {
            "path": str(clusters_output),
            "rows": len(all_clusters),
            "sha256": sha256_file(clusters_output),
        },
        "cards": {
            "path": str(args.output),
            "rows": len(cards),
            "sha256": sha256_file(args.output),
        },
        "audit": {
            "path": str(args.audit_output),
            "rows": len(audits),
            "sha256": sha256_file(args.audit_output),
        },
    }
    manifest = {
        "schema_version": "shopping-experience-distillation-manifest-v2",
        "created_at": args.created_at,
        "prompt_versions": {
            "segment_extraction": SEGMENT_EXTRACTION_PROMPT_VERSION,
            "cluster_distillation": CLUSTER_DISTILLATION_PROMPT_VERSION,
        },
        "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
        "sources": source_manifest,
        "forbidden_task_sources": forbidden_sources,
        "forbidden_task_overlap": [],
        "trajectory_count": len(inputs),
        "segmentless_trajectory_count": len(segmentless_trajectories),
        "segmentless_trajectory_ids": segmentless_trajectories,
        "segment_count": len(extractions),
        "cluster_count": len(all_clusters),
        "distilled_cluster_count": len(selected_clusters),
        "raw_candidate_count": len(raw_cards),
        "candidate_experience_count": len(cards),
        "previous_cards": (
            {
                "path": str(args.previous_cards),
                "rows": len(previous_cards),
                "sha256": sha256_file(args.previous_cards),
            }
            if args.previous_cards
            else None
        ),
        "outputs": outputs,
    }
    write_json_atomic(args.manifest, manifest, force=args.force)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
