#!/usr/bin/env python3
"""Build a validation-only store without bypassing production promotion gates."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from shopping_grpo.evaluation.artifacts import iter_jsonl, write_json_atomic, write_jsonl_atomic
from shopping_grpo.evaluation.manifest import sha256_file
from shopping_grpo.experience.contracts import finalize_experience_card, validate_experience_card
from shopping_grpo.experience.store import ExperienceStore, build_store_manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-store", type=Path)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--store-id", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        datetime.fromisoformat(args.created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit("--created-at must be ISO-8601") from exc

    base_cards = (
        [validate_experience_card(card) for card in iter_jsonl(args.base_store)]
        if args.base_store
        else []
    )
    if any(card["status"] != "active" for card in base_cards):
        raise SystemExit("--base-store may contain only active cards")
    candidates = [
        validate_experience_card(card) for card in iter_jsonl(args.candidates)
    ]
    requested = set(args.candidate_id)
    if requested:
        available = {card["experience_id"] for card in candidates}
        missing = sorted(requested - available)
        if missing:
            raise SystemExit(f"unknown --candidate-id values: {missing}")
        candidates = [
            card for card in candidates if card["experience_id"] in requested
        ]
    if not candidates:
        raise SystemExit("validation store requires at least one candidate")
    if any(card["status"] not in {"candidate", "validated"} for card in candidates):
        raise SystemExit("candidate input contains a non-candidate lifecycle status")
    temporary_active = []
    for card in candidates:
        promoted = dict(card)
        promoted["status"] = "active"
        temporary_active.append(finalize_experience_card(promoted))
    cards = ExperienceStore(base_cards + temporary_active).active_cards
    write_jsonl_atomic(args.output, cards, force=args.force)
    manifest = build_store_manifest(
        store_id=args.store_id,
        store_role="validation_only",
        cards_path=args.output,
        rows=len(cards),
        created_at=args.created_at,
        retriever={"backend": "lexical"},
    )
    manifest["validation_only"] = {
        "base_store": (
            {
                "path": str(args.base_store),
                "sha256": sha256_file(args.base_store),
            }
            if args.base_store
            else None
        ),
        "candidates": {
            "path": str(args.candidates),
            "sha256": sha256_file(args.candidates),
            "selected_experience_ids": sorted(
                card["experience_id"] for card in candidates
            ),
        },
        "production_eligible": False,
    }
    write_json_atomic(args.manifest, manifest, force=args.force)


if __name__ == "__main__":
    main()
