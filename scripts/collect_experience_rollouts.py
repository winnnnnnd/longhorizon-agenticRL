#!/usr/bin/env python3
"""Collect resumable multi-rollout trajectories for offline experience distillation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import signal

from shopping_grpo.evaluation.artifacts import iter_jsonl, load_json, write_json_atomic
from shopping_grpo.evaluation.manifest import sha256_file
from shopping_grpo.evaluation.rollout import (
    OpenAIChatClient,
    collect_tasks,
    load_tasks,
    rollout_interrupted,
)
from shopping_grpo.experience.contracts import SCOPE_CONTRACT, canonical_sha256
from shopping_grpo.experience.factory import build_experience_components


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--run-config",
        type=Path,
        help="不可变恢复配置；默认在 trajectory 文件旁生成 .run.json。",
    )
    parser.add_argument(
        "--held-out-tasks",
        type=Path,
        default=Path("data/evaluation/tasks.jsonl"),
    )
    parser.add_argument("--actor-role", choices=("teacher", "student"), required=True)
    parser.add_argument("--actor-revision", required=True)
    parser.add_argument("--attempts-per-task", type=int, default=4)
    parser.add_argument("--sampling-seed-base", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--experience-config", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--model", required=True)
    parser.add_argument("--llm-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--context-window", type=int, default=24576)
    parser.add_argument("--context-safety-margin", type=int, default=512)
    parser.add_argument("--context-compaction", action="store_true")
    parser.add_argument("--observation-token-budget", type=int, default=1536)
    parser.add_argument("--observation-detail-token-budget", type=int, default=4096)
    parser.add_argument("--observation-generic-token-budget", type=int, default=768)
    parser.add_argument("--observation-search-top-k", type=int, default=20)
    parser.add_argument("--force-manifest", action="store_true")
    return parser.parse_args()


def _validate_args(args):
    if args.attempts_per_task < 1:
        raise SystemExit("--attempts-per-task must be at least one")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least one")
    if not args.llm_base_url:
        raise SystemExit("--llm-base-url or OPENAI_BASE_URL is required")
    if not args.api_key:
        raise SystemExit("--api-key or OPENAI_API_KEY is required")
    if args.context_window <= args.max_tokens + args.context_safety_margin:
        raise SystemExit("actor context window is smaller than its protected budget")
    if args.experience_config and not args.context_window:
        raise SystemExit("experience injection requires exact actor context counting")


def _task_ids(path: Path) -> set[int]:
    return {int(row["task_id"]) for row in iter_jsonl(path)}


def _freeze_run_config(args, tasks, components) -> tuple[Path, dict]:
    path = args.run_config or Path(str(args.output) + ".run.json")
    config = {
        "schema_version": "shopping-experience-rollout-run-config-v1",
        "actor": {
            "role": args.actor_role,
            "model": args.model,
            "revision": args.actor_revision,
            "llm_base_url": args.llm_base_url,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "tasks": {
            "path": str(args.tasks),
            "sha256": sha256_file(args.tasks),
            "task_ids_hash": canonical_sha256([task["task_id"] for task in tasks]),
            "held_out_path": str(args.held_out_tasks),
            "held_out_sha256": sha256_file(args.held_out_tasks),
        },
        "rollout": {
            "shopsim_base_url": args.base_url,
            "attempts_per_task": args.attempts_per_task,
            "sampling_seeds": [
                args.sampling_seed_base + index
                for index in range(args.attempts_per_task)
            ],
            "max_steps": args.max_steps,
            "max_tokens": args.max_tokens,
            "context_window": args.context_window,
            "context_safety_margin": args.context_safety_margin,
            "context_compaction": args.context_compaction or components is not None,
            "semantic_compaction": components is not None,
            "observation_token_budget": args.observation_token_budget,
            "observation_detail_token_budget": args.observation_detail_token_budget,
            "observation_generic_token_budget": args.observation_generic_token_budget,
            "observation_search_top_k": args.observation_search_top_k,
        },
        "experience": components.runtime.public_manifest() if components else None,
    }
    if path.exists():
        if load_json(path) != config:
            raise SystemExit("existing rollout run config differs from this invocation")
    else:
        if args.output.exists():
            raise SystemExit("refusing to resume trajectories without a frozen run config")
        write_json_atomic(path, config)
    return path, config


def main():
    args = parse_args()
    _validate_args(args)
    tasks = load_tasks(args.tasks)
    if len(tasks) != len({task["task_id"] for task in tasks}):
        raise SystemExit("--tasks contains duplicate task IDs")
    if args.limit is not None:
        tasks = tasks[: args.limit]
    held_out = _task_ids(args.held_out_tasks)
    overlap = sorted({task["task_id"] for task in tasks} & held_out)
    if overlap:
        raise SystemExit(f"experience rollout tasks overlap held-out evaluation: {overlap}")

    components = (
        build_experience_components(args.experience_config)
        if args.experience_config
        else None
    )
    run_config_path, run_config = _freeze_run_config(args, tasks, components)
    client = OpenAIChatClient(
        model=args.model,
        base_url=args.llm_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        context_window=args.context_window,
        context_safety_margin=args.context_safety_margin,
        context_compaction_enable=args.context_compaction or components is not None,
        context_compactor=(components.context_compactor if components else None),
        observation_token_budget=args.observation_token_budget,
        observation_detail_token_budget=args.observation_detail_token_budget,
        observation_generic_token_budget=args.observation_generic_token_budget,
        observation_search_top_k=args.observation_search_top_k,
    )
    signal.signal(signal.SIGTERM, rollout_interrupted)
    signal.signal(signal.SIGINT, rollout_interrupted)
    collect_tasks(
        tasks,
        client=client,
        output_path=args.output,
        base_url=args.base_url,
        max_steps=args.max_steps,
        attempts_per_task=args.attempts_per_task,
        sampling_seeds=[
            args.sampling_seed_base + index
            for index in range(args.attempts_per_task)
        ],
        experience_runtime=(components.runtime if components else None),
    )

    rows = list(iter_jsonl(args.output))
    status_counts = Counter(str(row.get("status") or "unknown") for row in rows)
    manifest = {
        "schema_version": "shopping-experience-rollout-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "actor": {
            "role": args.actor_role,
            "model": args.model,
            "revision": args.actor_revision,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "tasks": {
            "path": str(args.tasks),
            "sha256": sha256_file(args.tasks),
            "task_ids_hash": canonical_sha256([task["task_id"] for task in tasks]),
            "count": len(tasks),
            "held_out_path": str(args.held_out_tasks),
            "held_out_sha256": sha256_file(args.held_out_tasks),
            "held_out_overlap": [],
        },
        "rollout": {
            "attempts_per_task": args.attempts_per_task,
            "sampling_seeds": [
                args.sampling_seed_base + index
                for index in range(args.attempts_per_task)
            ],
            "max_steps": args.max_steps,
            "max_tokens": args.max_tokens,
            "context_window": args.context_window,
            "context_safety_margin": args.context_safety_margin,
            "context_compaction": args.context_compaction or components is not None,
            "semantic_compaction": components is not None,
            "observation_token_budget": args.observation_token_budget,
        },
        "experience": components.runtime.public_manifest() if components else None,
        "run_config": {
            "path": str(run_config_path),
            "sha256": sha256_file(run_config_path),
            "content_hash": canonical_sha256(run_config),
        },
        "contract": SCOPE_CONTRACT,
        "output": {
            "path": str(args.output),
            "rows": len(rows),
            "sha256": sha256_file(args.output),
            "status_counts": dict(sorted(status_counts.items())),
        },
    }
    write_json_atomic(args.manifest, manifest, force=args.force_manifest)


if __name__ == "__main__":
    main()
