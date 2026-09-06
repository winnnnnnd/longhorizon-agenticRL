#!/usr/bin/env python3
"""在固定 ShopSimulator benchmark 上评测 OpenAI-compatible 本地或远端模型。"""

import argparse
import json
from pathlib import Path

from shopping_grpo.evaluation.summary import summarize_trajectories
from shopping_grpo.evaluation.rollout import OpenAIChatClient, collect_tasks, load_tasks
from shopping_grpo.experience.factory import build_experience_components


def parse_args():
    parser = argparse.ArgumentParser(description="评测 Base、SFT 或 GRPO Shopping Agent")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="原始评测轨迹 JSONL")
    parser.add_argument("--summary", type=Path, required=True, help="汇总指标 JSON")
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--actor-revision",
        help="经验实验必须填写不可漂移的 checkpoint/revision/hash。",
    )
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--api-key", required=True, help="本地 vLLM 可传 EMPTY")
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="单次模型生成上限；防止未调用工具时耗尽完整上下文。",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=24576,
        help="上下文窗口；传 0 禁用 vLLM /tokenize 依赖。",
    )
    parser.add_argument("--context-safety-margin", type=int, default=512)
    parser.add_argument(
        "--context-compaction",
        action="store_true",
        help="上下文接近上限时压缩较早的交互；默认关闭。",
    )
    parser.add_argument(
        "--observation-token-budget",
        type=int,
        default=1536,
        help="Observation token 预算；传 0 禁用 vLLM /tokenize 依赖。",
    )
    parser.add_argument("--observation-detail-token-budget", type=int, default=4096)
    parser.add_argument("--observation-generic-token-budget", type=int, default=768)
    parser.add_argument("--observation-search-top-k", type=int, default=20)
    parser.add_argument(
        "--experience-config",
        type=Path,
        help=(
            "启用 Frozen Agent 外部经验；同时强制启用 DSV4 Flash 语义上下文压缩。"
        ),
    )
    return parser.parse_args()


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    args = parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps 必须为正数")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens 必须为正数")
    if args.context_window < 0:
        raise SystemExit("--context-window 不能为负数")
    if args.context_window and args.context_window <= args.max_tokens + args.context_safety_margin:
        raise SystemExit("--context-window 必须大于 --max-tokens 与安全余量之和")
    if args.observation_token_budget < 0:
        raise SystemExit("--observation-token-budget 不能为负数")
    if args.experience_config and not args.context_window:
        raise SystemExit("经验注入要求非零 --context-window 和精确 tokenizer 计数")
    if args.experience_config and not args.actor_revision:
        raise SystemExit("经验实验要求显式 --actor-revision")
    experience_components = (
        build_experience_components(
            args.experience_config,
            allow_validation_only=False,
        )
        if args.experience_config
        else None
    )
    tasks = load_tasks(args.benchmark)
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
        context_compaction_enable=(
            args.context_compaction or experience_components is not None
        ),
        context_compactor=(
            experience_components.context_compactor
            if experience_components is not None
            else None
        ),
        observation_token_budget=args.observation_token_budget,
        observation_detail_token_budget=args.observation_detail_token_budget,
        observation_generic_token_budget=args.observation_generic_token_budget,
        observation_search_top_k=args.observation_search_top_k,
    )
    collect_tasks(
        tasks,
        client=client,
        output_path=args.output,
        base_url=args.base_url,
        max_steps=args.max_steps,
        experience_runtime=(
            experience_components.runtime
            if experience_components is not None
            else None
        ),
    )
    summary = summarize_trajectories(
        [task["task_id"] for task in tasks], _read_jsonl(args.output)
    )
    summary["protocol"] = {
        "benchmark": str(args.benchmark),
        "model": args.model,
        "actor_revision": args.actor_revision,
        "reward_contract": "shopsimulator-reward-v3",
        "max_steps": args.max_steps,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "context_window": args.context_window,
        "context_safety_margin": args.context_safety_margin,
        "context_compaction": args.context_compaction or experience_components is not None,
        "context_compaction_strategy": (
            "deepseek_v4_flash_grounded"
            if experience_components is not None
            else (
                "drop_old_complete_groups"
                if args.context_compaction
                else "disabled"
            )
        ),
        "observation_token_budget": args.observation_token_budget,
        "observation_detail_token_budget": args.observation_detail_token_budget,
        "observation_generic_token_budget": args.observation_generic_token_budget,
        "observation_search_top_k": args.observation_search_top_k,
        "experience": (
            experience_components.runtime.public_manifest()
            if experience_components is not None
            else None
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
