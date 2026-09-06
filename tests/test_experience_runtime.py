import json
import unittest

from shopping_grpo.experience.contracts import (
    EXPERIENCE_CARD_VERSION,
    ExperienceBundle,
    SCOPE_CONTRACT,
    finalize_experience_card,
)
from shopping_grpo.experience.runtime import ExperienceRuntime


def experience_card():
    return finalize_experience_card(
        {
            "schema_version": EXPERIENCE_CARD_VERSION,
            "experience_id": "exp-search-distinct-query-001",
            "revision": 1,
            "status": "active",
            "type": "search_strategy",
            "phase": "search",
            "trigger": {
                "events": ["task_start"],
                "page_types": [],
                "required_constraint_tags": [],
                "predicates": [],
            },
            "guidance": ["查询保留品类和最有区分度的条件。"],
            "anti_patterns": [],
            "verification_checks": ["确认新查询带来不同候选。"],
            "scope": {**SCOPE_CONTRACT, "categories": []},
            "evidence": {
                "source_kind": "curated_teacher_gold",
                "supporting_trajectory_ids": ["teacher-1"],
                "contradicting_trajectory_ids": [],
                "support_count": 1,
            },
            "provenance": {
                "extractor": "deepseek-v4-flash",
                "extractor_revision": "test-revision",
                "prompt_version": "experience-distill-v1",
                "created_at": "2026-09-06T00:00:00+00:00",
            },
            "supersedes": None,
            "content_hash": "pending",
        }
    )


class FakeStore:
    manifest = {"store_id": "test-store"}


class FakeRetriever:
    backend = "lexical"
    top_k = 3
    minimum_score = 0.0
    store = FakeStore()

    def __init__(self):
        self.calls = []

    def retrieve(self, *, state, event, injection_counts):
        self.calls.append((state, event, dict(injection_counts)))
        card = experience_card()
        return ExperienceBundle(
            recall_event=event,
            state_signature_hash=state.signature(event),
            eligible_experience_ids=(card["experience_id"],),
            cards=(card,),
            scores={card["experience_id"]: 0.9},
        )


def count_tokens(messages, tools):
    return len(json.dumps(messages, ensure_ascii=False)) // 4


class ExperienceRuntimeTest(unittest.TestCase):
    def test_injection_is_ephemeral_and_task_start_recall_is_cached(self):
        retriever = FakeRetriever()
        runtime = ExperienceRuntime(retriever, max_experience_tokens=500)
        session = runtime.start_session(task={"task_id": 7}, max_steps=35)
        messages = [
            {"role": "system", "content": "fixed rules"},
            {"role": "user", "content": "Instruction: 购买可水洗水彩笔"},
        ]
        trajectory = {
            "task_id": 7,
            "initial_result": {"instruction": "Instruction: 购买可水洗水彩笔"},
            "steps": [],
            "blocked_tool_calls": [],
        }

        request, event = session.prepare_request(
            messages=messages,
            tools=[],
            trajectory=trajectory,
            latest_observation="Instruction: 购买可水洗水彩笔",
            count_tokens=count_tokens,
        )
        repeated, repeated_event = session.prepare_request(
            messages=messages,
            tools=[],
            trajectory=trajectory,
            latest_observation="Instruction: 购买可水洗水彩笔",
            count_tokens=count_tokens,
        )

        self.assertNotIn("EXPERIENCE_GUIDANCE", messages[0]["content"])
        self.assertIn("EXPERIENCE_GUIDANCE", request[0]["content"])
        self.assertEqual(request, repeated)
        self.assertEqual(event["recall_event"], "task_start")
        self.assertTrue(repeated_event["cache_reused"])
        self.assertEqual(len(retriever.calls), 1)


if __name__ == "__main__":
    unittest.main()
