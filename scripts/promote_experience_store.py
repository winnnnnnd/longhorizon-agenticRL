#!/usr/bin/env python3
"""Create a frozen active experience store only from two-gate promotion decisions."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from shopping_grpo.evaluation.artifacts import iter_jsonl, write_json_atomic, write_jsonl_atomic
from shopping_grpo.evaluation.manifest import sha256_file
from shopping_grpo.experience.promotion import apply_promotion_decisions
from shopping_grpo.experience.store import build_store_manifest


def parse_args():
    parser = argparse.ArgumentParser(description="依据冻结验证决策构建 Active Experience Store")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--store-id", required=True)
    parser.add_argument(
        "--created-at",
        required=True,
        help="固定 ISO-8601 时间；重建相同 Store 时必须复用同一值。",
    )
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--retrieval-backend", choices=("lexical", "embedding"), default="lexical")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        datetime.fromisoformat(args.created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit("--created-at must be ISO-8601") from exc
    cards = apply_promotion_decisions(
        iter_jsonl(args.candidates),
        iter_jsonl(args.decisions),
        expected_validation_manifest_hash=sha256_file(args.validation_manifest),
    )
    active = [card for card in cards if card["status"] == "active"]
    write_jsonl_atomic(args.output, active, force=args.force)
    manifest = build_store_manifest(
        store_id=args.store_id,
        cards_path=args.output,
        rows=len(active),
        created_at=args.created_at,
        embeddings_path=args.embeddings,
        retriever={"backend": args.retrieval_backend},
    )
    manifest["promotion"] = {
        "candidates": {
            "path": str(args.candidates),
            "sha256": sha256_file(args.candidates),
        },
        "decisions": {
            "path": str(args.decisions),
            "sha256": sha256_file(args.decisions),
        },
        "validation_manifest": {
            "path": str(args.validation_manifest),
            "sha256": sha256_file(args.validation_manifest),
        },
        "active_rows": len(active),
        "rejected_rows": sum(card["status"] == "rejected" for card in cards),
    }
    write_json_atomic(args.manifest, manifest, force=args.force)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
