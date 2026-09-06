"""Trajectory-local experience retrieval and ephemeral request preparation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy

from shopping_grpo.experience.contracts import (
    EXPERIENCE_INJECTION_VERSION,
    EXPERIENCE_RETRIEVER_VERSION,
    canonical_sha256,
)
from shopping_grpo.experience.events import build_state_view, detect_recall_event
from shopping_grpo.experience.injection import inject_experience_bundle


class ExperienceRuntime:
    """Immutable shared dependencies; all mutable recall state lives in a session."""

    requires_semantic_compaction = True

    def __init__(
        self,
        retriever,
        *,
        max_experience_tokens: int = 500,
        semantic_compaction_manifest: Mapping | None = None,
    ):
        if int(max_experience_tokens) < 1:
            raise ValueError("max_experience_tokens must be positive")
        self.retriever = retriever
        self.max_experience_tokens = int(max_experience_tokens)
        self.store_manifest_hash = canonical_sha256(retriever.store.manifest)
        self.semantic_compaction_manifest = deepcopy(
            dict(semantic_compaction_manifest or {})
        )

    def public_manifest(self) -> dict:
        return {
            "retriever_version": EXPERIENCE_RETRIEVER_VERSION,
            "retrieval_backend": self.retriever.backend,
            "top_k": self.retriever.top_k,
            "minimum_score": self.retriever.minimum_score,
            "injection_prompt_version": EXPERIENCE_INJECTION_VERSION,
            "max_experience_tokens": self.max_experience_tokens,
            "store_manifest_hash": self.store_manifest_hash,
            "semantic_compaction": deepcopy(self.semantic_compaction_manifest),
        }

    def start_session(self, *, task: Mapping, max_steps: int) -> "ExperienceSession":
        return ExperienceSession(self, task=task, max_steps=max_steps)


class ExperienceSession:
    def __init__(self, runtime: ExperienceRuntime, *, task: Mapping, max_steps: int):
        self.runtime = runtime
        self.task = deepcopy(dict(task))
        self.max_steps = int(max_steps)
        self.bundle_by_signature = {}
        self.active_bundle = None
        self.last_phase = None
        self.injection_counts = Counter()

    def header(self) -> dict:
        return {"store": self.runtime.public_manifest(), "events": []}

    def prepare_request(
        self,
        *,
        messages,
        tools,
        trajectory: Mapping,
        latest_observation: str,
        count_tokens,
    ) -> tuple[list[dict], dict]:
        if count_tokens is None:
            raise ValueError("experience injection requires the actor's exact token counter")
        state = build_state_view(
            task=self.task,
            trajectory=trajectory,
            latest_observation=latest_observation,
            max_steps=self.max_steps,
        )
        event = detect_recall_event(state)
        cache_reused = False
        retrieval_error = None
        if event is not None:
            signature = state.signature(event)
            if signature in self.bundle_by_signature:
                cache_reused = True
                self.active_bundle = self.bundle_by_signature[signature]
            else:
                try:
                    self.active_bundle = self.runtime.retriever.retrieve(
                        state=state,
                        event=event,
                        injection_counts=self.injection_counts,
                    )
                except Exception as exc:
                    retrieval_error = f"{exc.__class__.__name__}:{exc}"
                    self.active_bundle = None
                self.bundle_by_signature[signature] = self.active_bundle
        elif self.last_phase is not None and state.phase != self.last_phase:
            self.active_bundle = None
        self.last_phase = state.phase

        bundle = self.active_bundle
        state_audit = {
            "phase": state.phase,
            "page_type": state.page_type,
            "constraint_tags": list(state.constraint_tags),
            "predicates": list(state.predicates),
            "current_product_id": state.current_product_id,
            "candidate_set_hash": state.candidate_set_hash,
            "selected_options": list(state.selected_options),
            "last_tool_name": state.last_tool_name,
            "last_guard_reason": state.last_guard_reason,
            "executed_steps": state.executed_steps,
            "remaining_steps": state.remaining_steps,
        }
        if bundle is None:
            request_messages = [deepcopy(dict(message)) for message in messages]
            audit = {
                "step_index": len(trajectory.get("steps") or []),
                "recall_event": event,
                "state_signature_hash": state.signature(event) if event else None,
                "eligible_experience_ids": [],
                "selected_experience_ids": [],
                "scores": {},
                "experience_tokens": 0,
                "bundle_hash": None,
                "cache_reused": cache_reused,
                "retrieval_error": retrieval_error,
                "state": state_audit,
            }
            return request_messages, audit

        injection = inject_experience_bundle(
            messages,
            tools,
            bundle,
            count_tokens=count_tokens,
            max_experience_tokens=self.runtime.max_experience_tokens,
        )
        selected = set(injection.selected_experience_ids)
        for experience_id in selected:
            self.injection_counts[experience_id] += 1
        audit = {
            "step_index": len(trajectory.get("steps") or []),
            "recall_event": event or bundle.recall_event,
            "state_signature_hash": bundle.state_signature_hash,
            "eligible_experience_ids": list(bundle.eligible_experience_ids),
            "selected_experience_ids": list(injection.selected_experience_ids),
            "scores": {
                experience_id: float(bundle.scores[experience_id])
                for experience_id in injection.selected_experience_ids
            },
            "experience_tokens": injection.experience_tokens,
            "bundle_hash": injection.bundle_hash,
            "cache_reused": cache_reused or event is None,
            "retrieval_error": retrieval_error,
            "state": state_audit,
        }
        return injection.messages, audit
