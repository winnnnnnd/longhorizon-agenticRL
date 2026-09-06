import unittest

from shopping_grpo.experience.contracts import (
    EXPERIENCE_CARD_VERSION,
    ExperienceContractError,
    SCOPE_CONTRACT,
    finalize_experience_card,
)
from shopping_grpo.experience.store import ExperienceStore


def experience_card(**updates):
    payload = {
        "schema_version": EXPERIENCE_CARD_VERSION,
        "experience_id": "exp-search-distinct-query-001",
        "revision": 1,
        "status": "active",
        "type": "search_strategy",
        "phase": "search",
        "trigger": {
            "events": ["task_start", "search_stagnation"],
            "page_types": [],
            "required_constraint_tags": [],
            "predicates": [],
        },
        "guidance": ["查询保留品类和最有区分度的需求条件。"],
        "anti_patterns": ["不要只做同义改写。"],
        "verification_checks": ["新查询应带来新的候选集合。"],
        "scope": {**SCOPE_CONTRACT, "categories": []},
        "evidence": {
            "source_kind": "curated_teacher_gold",
            "supporting_trajectory_ids": ["teacher-1", "teacher-2"],
            "contradicting_trajectory_ids": [],
            "support_count": 2,
        },
        "provenance": {
            "extractor": "deepseek-v4-flash",
            "extractor_revision": "deepseek-v4-flash",
            "prompt_version": "experience-distill-v1",
            "created_at": "2026-09-06T00:00:00+00:00",
        },
        "supersedes": None,
        "content_hash": "pending",
    }
    payload.update(updates)
    return finalize_experience_card(payload)


class ExperienceContractTest(unittest.TestCase):
    def test_finalized_card_is_stable_and_loads_as_active(self):
        card = experience_card()
        repeated = finalize_experience_card(card)
        store = ExperienceStore([card])

        self.assertEqual(card["content_hash"], repeated["content_hash"])
        self.assertEqual(store.active_cards[0]["experience_id"], card["experience_id"])

    def test_semantic_content_rejects_product_ids(self):
        with self.assertRaisesRegex(ExperienceContractError, "product ID"):
            experience_card(guidance=["打开商品 123456789012 后直接购买。"])

    def test_support_count_must_match_ids(self):
        evidence = experience_card()["evidence"]
        evidence["support_count"] = 1
        with self.assertRaisesRegex(ExperienceContractError, "support_count"):
            experience_card(evidence=evidence)

    def test_unknown_top_level_field_is_rejected(self):
        with self.assertRaisesRegex(ExperienceContractError, "unexpected fields"):
            experience_card(gold_purchase={"asin": "123456789012"})


if __name__ == "__main__":
    unittest.main()
