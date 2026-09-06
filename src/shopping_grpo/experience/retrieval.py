"""Deterministic hard filtering plus lexical or precomputed-vector ranking."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
import math
import re

from shopping_grpo.experience.contracts import (
    RECALL_EVENT_PHASES,
    ExperienceBundle,
    ExperienceStateView,
)
from shopping_grpo.experience.store import ExperienceStore


_CJK = re.compile(r"[\u3400-\u9fff]+")
_WORD = re.compile(r"[a-z0-9_]+")


def lexical_features(text: object) -> Counter[str]:
    folded = str(text or "").casefold()
    features = Counter(_WORD.findall(folded))
    for segment in _CJK.findall(folded):
        if len(segment) == 1:
            features[segment] += 1
        else:
            features.update(segment[index : index + 2] for index in range(len(segment) - 1))
    return features


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("query and experience embedding dimensions differ")
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _weighted_jaccard(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    intersection = sum(min(left[key], right[key]) for key in keys)
    union = sum(max(left[key], right[key]) for key in keys)
    return intersection / union if union else 0.0


def _card_text(card: Mapping) -> str:
    trigger = card["trigger"]
    parts = [card["type"], card["phase"]]
    parts.extend(trigger.get("required_constraint_tags") or [])
    parts.extend(card.get("guidance") or [])
    parts.extend(card.get("anti_patterns") or [])
    parts.extend(card.get("verification_checks") or [])
    parts.extend((card.get("scope") or {}).get("categories") or [])
    return "\n".join(str(part) for part in parts)


def _state_text(state: ExperienceStateView, event: str) -> str:
    return "\n".join(
        (
            state.public_query,
            event,
            state.phase,
            state.page_type,
            " ".join(state.constraint_tags),
            " ".join(state.unverified_constraints),
            " ".join(state.predicates),
        )
    )


def _eligible(card: Mapping, state: ExperienceStateView, event: str) -> bool:
    trigger = card["trigger"]
    if event not in trigger["events"]:
        return False
    if card["phase"] not in RECALL_EVENT_PHASES[event]:
        return False
    if trigger["page_types"] and state.page_type not in trigger["page_types"]:
        return False
    if trigger["required_constraint_tags"] and not set(
        trigger["required_constraint_tags"]
    ).issubset(state.constraint_tags):
        return False
    if trigger["predicates"] and not set(trigger["predicates"]).issubset(
        state.predicates
    ):
        return False
    categories = card["scope"].get("categories") or []
    if categories and not any(
        category.casefold() in state.public_query.casefold() for category in categories
    ):
        return False
    return True


class ExperienceRetriever:
    def __init__(
        self,
        store: ExperienceStore,
        *,
        top_k: int = 3,
        backend: str = "lexical",
        query_embedder: Callable[[str], tuple[float, ...]] | None = None,
        minimum_score: float = 0.0,
    ):
        if int(top_k) < 1:
            raise ValueError("experience top_k must be positive")
        if backend not in {"lexical", "embedding"}:
            raise ValueError("experience retrieval backend must be lexical or embedding")
        if backend == "embedding" and query_embedder is None:
            raise ValueError("embedding retrieval requires a query_embedder")
        self.store = store
        self.top_k = int(top_k)
        self.backend = backend
        self.query_embedder = query_embedder
        self.minimum_score = float(minimum_score)
        if backend == "embedding":
            missing = [
                card["experience_id"]
                for card in store.active_cards
                if store.embedding_for(card) is None
            ]
            if missing:
                raise ValueError(
                    "embedding store is missing active experiences: "
                    + ", ".join(sorted(missing))
                )

    def retrieve(
        self,
        *,
        state: ExperienceStateView,
        event: str,
        injection_counts: Mapping[str, int] | None = None,
    ) -> ExperienceBundle:
        candidates = [
            card for card in self.store.active_cards if _eligible(card, state, event)
        ]
        eligible_ids = tuple(card["experience_id"] for card in candidates)
        query_text = _state_text(state, event)
        query_vector = (
            self.query_embedder(query_text) if self.backend == "embedding" else None
        )
        query_lexical = lexical_features(query_text)
        counts = injection_counts or {}
        ranked = []
        for card in candidates:
            if self.backend == "embedding":
                vector = self.store.embedding_for(card)
                if vector is None:
                    continue
                semantic = _cosine(tuple(query_vector), vector)
            else:
                semantic = _weighted_jaccard(
                    query_lexical, lexical_features(_card_text(card))
                )
            trigger = card["trigger"]
            score = semantic
            score += 0.20 if event in trigger["events"] else 0.0
            score += 0.10 if card["phase"] == state.phase else 0.0
            score += 0.05 * len(
                set(trigger["required_constraint_tags"]) & set(state.constraint_tags)
            )
            score -= min(0.15, 0.03 * int(counts.get(card["experience_id"], 0)))
            if score >= self.minimum_score:
                ranked.append((float(score), card))
        ranked.sort(key=lambda item: (-item[0], item[1]["experience_id"]))
        selected = ranked[: self.top_k]
        return ExperienceBundle(
            recall_event=event,
            state_signature_hash=state.signature(event),
            eligible_experience_ids=eligible_ids,
            cards=tuple(card for _, card in selected),
            scores={card["experience_id"]: score for score, card in selected},
        )
