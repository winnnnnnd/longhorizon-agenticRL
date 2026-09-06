"""Immutable experience-store loading with strict hash and revision checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import math
from pathlib import Path

from shopping_grpo.evaluation.artifacts import iter_jsonl, load_json
from shopping_grpo.evaluation.manifest import sha256_file
from shopping_grpo.experience.contracts import (
    EXPERIENCE_STORE_MANIFEST_VERSION,
    ExperienceContractError,
    SCOPE_CONTRACT,
    validate_experience_card,
)


class ExperienceStore:
    """A validated, read-only snapshot used for one complete actor run."""

    def __init__(
        self,
        cards: Iterable[Mapping],
        *,
        manifest: Mapping | None = None,
        embeddings: Mapping[tuple[str, int], tuple[float, ...]] | None = None,
    ):
        validated = [validate_experience_card(card) for card in cards]
        keys = [(card["experience_id"], card["revision"]) for card in validated]
        if len(keys) != len(set(keys)):
            raise ExperienceContractError("experience store contains duplicate ID/revision")
        hashes = [card["content_hash"] for card in validated]
        if len(hashes) != len(set(hashes)):
            raise ExperienceContractError("experience store contains duplicate semantic content")
        latest_revision = {}
        for card in validated:
            latest_revision[card["experience_id"]] = max(
                card["revision"], latest_revision.get(card["experience_id"], 0)
            )
        active = [
            card
            for card in validated
            if card["status"] == "active"
            and card["revision"] == latest_revision[card["experience_id"]]
        ]
        superseded = {card.get("supersedes") for card in active if card.get("supersedes")}
        if superseded.intersection(card["experience_id"] for card in active):
            raise ExperienceContractError("an active experience supersedes another active card")
        self._cards = tuple(deepcopy(validated))
        self._active = tuple(deepcopy(active))
        self._manifest = deepcopy(dict(manifest or {}))
        self._embeddings = dict(embeddings or {})

    @classmethod
    def load(
        cls,
        *,
        cards_path: str | Path,
        manifest_path: str | Path,
        embeddings_path: str | Path | None = None,
        allow_validation_only: bool = False,
    ) -> "ExperienceStore":
        cards_path = Path(cards_path)
        manifest = validate_store_manifest(load_json(manifest_path))
        if manifest["store_role"] == "validation_only" and not allow_validation_only:
            raise ExperienceContractError(
                "validation-only experience store is disabled for this run"
            )
        expected_cards = manifest["cards"]
        if cards_path.name != Path(expected_cards["path"]).name:
            raise ExperienceContractError("configured card path differs from store manifest")
        actual_hash = sha256_file(cards_path)
        if actual_hash != expected_cards["sha256"]:
            raise ExperienceContractError("experience card file hash mismatch")
        cards = list(iter_jsonl(cards_path))
        if len(cards) != expected_cards["rows"]:
            raise ExperienceContractError("experience card row count mismatch")
        embeddings = None
        if embeddings_path is not None:
            embeddings_path = Path(embeddings_path)
            expected = manifest.get("embeddings")
            if not isinstance(expected, Mapping):
                raise ExperienceContractError("store manifest does not declare embeddings")
            if sha256_file(embeddings_path) != expected.get("sha256"):
                raise ExperienceContractError("experience embedding file hash mismatch")
            embeddings = _load_embeddings(embeddings_path)
        return cls(cards, manifest=manifest, embeddings=embeddings)

    @property
    def cards(self) -> tuple[dict, ...]:
        return deepcopy(self._cards)

    @property
    def active_cards(self) -> tuple[dict, ...]:
        return deepcopy(self._active)

    @property
    def manifest(self) -> dict:
        return deepcopy(self._manifest)

    def embedding_for(self, card: Mapping) -> tuple[float, ...] | None:
        return self._embeddings.get((str(card["experience_id"]), int(card["revision"])))


def validate_store_manifest(manifest: object) -> dict:
    if not isinstance(manifest, Mapping):
        raise ExperienceContractError("experience store manifest must be an object")
    result = deepcopy(dict(manifest))
    if result.get("schema_version") != EXPERIENCE_STORE_MANIFEST_VERSION:
        raise ExperienceContractError("unsupported experience store manifest version")
    for field in ("store_id", "store_role", "created_at", "cards"):
        if field not in result:
            raise ExperienceContractError(f"experience store manifest is missing {field!r}")
    if result["store_role"] not in {"active", "validation_only"}:
        raise ExperienceContractError("experience store manifest role is unsupported")
    for field, expected in SCOPE_CONTRACT.items():
        if result.get(field) != expected:
            raise ExperienceContractError(
                f"experience store manifest {field} must equal {expected!r}"
            )
    cards = result["cards"]
    if not isinstance(cards, Mapping):
        raise ExperienceContractError("experience store manifest cards must be an object")
    if not isinstance(cards.get("rows"), int) or cards["rows"] < 0:
        raise ExperienceContractError("experience store manifest cards.rows is invalid")
    for field in ("path", "sha256"):
        if not isinstance(cards.get(field), str) or not cards[field]:
            raise ExperienceContractError(
                f"experience store manifest cards.{field} is required"
            )
    if len(cards["sha256"]) != 64:
        raise ExperienceContractError("experience store manifest card hash is invalid")
    return result


def _load_embeddings(path: Path) -> dict[tuple[str, int], tuple[float, ...]]:
    result = {}
    dimension = None
    for row in iter_jsonl(path):
        key = (str(row.get("experience_id") or ""), int(row.get("revision", 0)))
        vector = row.get("vector")
        if not key[0] or key[1] < 1 or not isinstance(vector, list) or not vector:
            raise ExperienceContractError("malformed experience embedding row")
        try:
            values = tuple(float(item) for item in vector)
        except (TypeError, ValueError) as exc:
            raise ExperienceContractError("experience embedding must be numeric") from exc
        if any(not math.isfinite(item) for item in values):
            raise ExperienceContractError("experience embedding must be finite")
        dimension = len(values) if dimension is None else dimension
        if len(values) != dimension:
            raise ExperienceContractError("experience embedding dimensions differ")
        if key in result:
            raise ExperienceContractError("duplicate experience embedding key")
        result[key] = values
    return result


def build_store_manifest(
    *,
    store_id: str,
    cards_path: str | Path,
    rows: int,
    created_at: str,
    embeddings_path: str | Path | None = None,
    retriever: Mapping | None = None,
    store_role: str = "active",
) -> dict:
    cards_path = Path(cards_path)
    manifest = {
        "schema_version": EXPERIENCE_STORE_MANIFEST_VERSION,
        "store_id": str(store_id),
        "store_role": str(store_role),
        "created_at": str(created_at),
        **SCOPE_CONTRACT,
        "cards": {
            "path": cards_path.name,
            "rows": int(rows),
            "sha256": sha256_file(cards_path),
        },
        "retriever": deepcopy(dict(retriever or {})),
    }
    if embeddings_path is not None:
        embeddings_path = Path(embeddings_path)
        manifest["embeddings"] = {
            "path": embeddings_path.name,
            "sha256": sha256_file(embeddings_path),
        }
    return validate_store_manifest(manifest)
