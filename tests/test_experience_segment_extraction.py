import json
import unittest

from shopping_grpo.experience.contracts import canonical_sha256, finalize_experience_card
from shopping_grpo.experience.promotion import merge_active_store
from shopping_grpo.experience.segment_extraction import (
    CLUSTER_DISTILLATION_VERSION,
    SEGMENT_EXTRACTION_VERSION,
    SegmentClusterDistiller,
    SequentialSegmentExtractor,
    apply_latest_card_revisions,
    cluster_segment_extractions,
    segment_cluster_key,
)
from shopping_grpo.experience.segmentation import build_trajectory_segments


def sample_trajectory(trajectory_id="student-1", task_id=1):
    return {
        "schema_version": "shopping-normalized-trajectory-v1",
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "actor_query": "购买适合儿童使用且可水洗的水彩笔",
        "events": [
            {
                "event_id": "e0001",
                "event_type": "tool_step",
                "assistant_text": "先检索核心需求",
                "tool_name": "search_products",
                "parameters": {"query": "儿童 可水洗 水彩笔"},
                "actor_visible_observation": (
                    "page_type: search_results\n1|1234567890|儿童水彩笔"
                ),
            },
            {
                "event_id": "e0002",
                "event_type": "tool_step",
                "assistant_text": "打开候选核验",
                "tool_name": "open_product",
                "parameters": {"asin": "1234567890"},
                "actor_visible_observation": (
                    "page_type: product_detail\navailable_options: 颜色"
                ),
            },
        ],
        "status": "done",
        "terminal_result": {
            "reward_detail": {
                "reward_type": "gold_purchase",
                "reward_valid": True,
            }
        },
    }


class FakeSequentialClient:
    model = "deepseek-v4-flash"
    model_revision = "deepseek-v4-flash-test"

    def __init__(self, *, leak=False):
        self.payloads = []
        self.leak = leak

    def complete_json(self, messages):
        payload = json.loads(messages[1]["content"])
        self.payloads.append(payload)
        segment = payload["segment"]
        is_search = segment["phase"] == "search"
        owned_event = segment["events"][0]["event_id"]
        prior = payload["prior_segment_knowledge"]
        return {
            "result": {
                "schema_version": SEGMENT_EXTRACTION_VERSION,
                "segment_id": segment["segment_id"],
                "phase": segment["phase"],
                "summary": (
                    "候选 1234567890 值得核验"
                    if self.leak
                    else ("检索得到候选" if is_search else "进入候选详情")
                ),
                "state_transition": {
                    "before": "尚未确认候选" if is_search else "已有候选",
                    "decision": "保留核心约束进行检索" if is_search else "进入详情",
                    "after": "获得候选" if is_search else "进入详情页",
                },
                "constraint_tags": ["category", "core_function"],
                "state_predicates": [],
                "action_pattern": "initial_search" if is_search else "open_candidate",
                "failure_type": "none",
                "local_outcome": "progress",
                "prior_knowledge_refs": (
                    [prior[-1]["knowledge_id"]] if prior else []
                ),
                "positive_pattern": "检索时保留关键约束",
                "anti_pattern": None,
                "carried_knowledge": [
                    {
                        "kind": "strategy",
                        "statement": (
                            "后续应核验核心功能" if is_search else "已进入候选详情阶段"
                        ),
                        "evidence_event_ids": [owned_event],
                    }
                ],
                "evidence_event_ids": [owned_event],
            },
            "metadata": {"request_index": len(self.payloads)},
        }


class FakeClusterClient:
    model = "deepseek-v4-flash"
    model_revision = "deepseek-v4-flash-test"

    def complete_json(self, messages):
        payload = json.loads(messages[1]["content"])
        members = payload["cluster"]["members"]
        return {
            "result": {
                "schema_version": CLUSTER_DISTILLATION_VERSION,
                "analysis": {
                    "shared_pattern": "保留核心约束的检索能产生有效候选",
                    "applicability": "处于检索阶段且尚未获得可靠候选",
                },
                "experience": {
                    "type": "search_strategy",
                    "phase": "search",
                    "trigger": {
                        "events": ["task_start", "search_stagnation"],
                        "page_types": [],
                        "required_constraint_tags": ["category"],
                        "predicates": [],
                    },
                    "guidance": ["检索词同时保留品类与区分度最高的核心约束。"],
                    "anti_patterns": ["不要重复没有带来新候选的同一查询。"],
                    "verification_checks": ["确认新结果集合包含可继续核验的候选。"],
                    "categories": [],
                    "supporting_segment_ids": [
                        member["segment_id"] for member in members
                    ],
                    "contradicting_segment_ids": [],
                },
            },
            "metadata": {"selected_members": len(members)},
        }


class SegmentExtractionTest(unittest.TestCase):
    def test_contiguous_phase_segments_keep_adjacent_context(self):
        segments = build_trajectory_segments(
            sample_trajectory(),
            source_kind="agent_rollout",
            actor_role="student",
            context_events=1,
        )

        self.assertEqual([row["phase"] for row in segments], ["search", "candidate_screening"])
        self.assertEqual(segments[0]["context_after"][0]["event_id"], "e0002")
        self.assertEqual(segments[1]["context_before"][0]["event_id"], "e0001")
        self.assertIn("candidate_new", segments[1]["observed_state_predicates"])

    def test_sequential_extraction_passes_prior_knowledge_and_owns_cluster_key(self):
        client = FakeSequentialClient()
        extractions, audits = SequentialSegmentExtractor(
            client,
            context_events=1,
        ).extract_trajectory(
            sample_trajectory(),
            actor_role="student",
            source_kind="agent_rollout",
        )

        self.assertEqual(len(extractions), 2)
        self.assertEqual(client.payloads[0]["prior_segment_knowledge"], [])
        self.assertEqual(
            client.payloads[1]["prior_segment_knowledge"][0]["knowledge_id"],
            "student-1:seg0001:k001",
        )
        self.assertEqual(
            extractions[1]["prior_knowledge_refs"],
            ["student-1:seg0001:k001"],
        )
        self.assertIn("candidate_new", extractions[1]["state_predicates"])
        self.assertEqual(
            extractions[1]["cluster_key_hash"],
            canonical_sha256(segment_cluster_key(extractions[1])),
        )
        self.assertEqual(len(audits), 2)

    def test_segment_extraction_rejects_task_specific_product_id(self):
        with self.assertRaisesRegex(ValueError, "product ID"):
            SequentialSegmentExtractor(FakeSequentialClient(leak=True)).extract_trajectory(
                sample_trajectory(),
                actor_role="student",
                source_kind="agent_rollout",
            )

    def test_exact_six_dimension_cluster_and_card_lineage(self):
        first = SequentialSegmentExtractor(FakeSequentialClient()).extract_trajectory(
            sample_trajectory("student-1", 1),
            actor_role="student",
            source_kind="agent_rollout",
        )[0][0]
        second = dict(first)
        second.update(
            {
                "segment_id": "student-2:seg0001",
                "trajectory_id": "student-2",
                "task_id": 2,
            }
        )
        second["carried_knowledge"] = []
        second["cluster_key"] = segment_cluster_key(second)
        second["cluster_key_hash"] = canonical_sha256(second["cluster_key"])
        different = dict(second)
        different.update(
            {
                "segment_id": "student-3:seg0001",
                "trajectory_id": "student-3",
                "task_id": 3,
                "failure_type": "search_stagnation",
            }
        )
        different["cluster_key"] = segment_cluster_key(different)
        different["cluster_key_hash"] = canonical_sha256(different["cluster_key"])

        clusters = cluster_segment_extractions([first, second, different])
        self.assertEqual(sorted(cluster["member_count"] for cluster in clusters), [1, 2])
        shared = next(cluster for cluster in clusters if cluster["member_count"] == 2)
        self.assertEqual(
            set(shared["cluster_key"]),
            {
                "phase",
                "failure_type",
                "constraint_tags",
                "state_predicates",
                "action_pattern",
                "terminal_outcome",
            },
        )

        card, audit = SegmentClusterDistiller(
            FakeClusterClient(),
            created_at="2026-09-14T00:00:00+00:00",
        ).distill_cluster(shared)
        self.assertEqual(card["evidence"]["support_count"], 2)
        self.assertEqual(
            {row["segment_id"] for row in audit["lineage"]},
            {"student-1:seg0001", "student-2:seg0001"},
        )

    def test_latest_revision_supersedes_same_experience_id(self):
        extraction = SequentialSegmentExtractor(FakeSequentialClient()).extract_trajectory(
            sample_trajectory(),
            actor_role="student",
            source_kind="agent_rollout",
        )[0][0]
        cluster = cluster_segment_extractions([extraction])[0]
        previous, _ = SegmentClusterDistiller(
            FakeClusterClient(),
            created_at="2026-09-13T00:00:00+00:00",
        ).distill_cluster(cluster)
        new_card = dict(previous)
        new_card["guidance"] = ["先保留品类，再加入一个可核验的关键约束。"]
        new_card["evidence"] = {
            **previous["evidence"],
            "supporting_trajectory_ids": ["student-new"],
            "support_count": 1,
        }
        new_card = finalize_experience_card(new_card)

        updated = apply_latest_card_revisions([new_card], [previous])[0]
        self.assertEqual(updated["revision"], 2)
        self.assertEqual(
            updated["supersedes"],
            f"{previous['experience_id']}@1",
        )
        self.assertEqual(
            set(updated["evidence"]["supporting_trajectory_ids"]),
            {"student-1", "student-new"},
        )

        old_active = finalize_experience_card({**previous, "status": "active"})
        rejected = finalize_experience_card({**updated, "status": "rejected"})
        kept = merge_active_store([old_active], [rejected])
        self.assertEqual(kept[0]["revision"], 1)

        new_active = finalize_experience_card({**updated, "status": "active"})
        replaced = merge_active_store([old_active], [new_active])
        self.assertEqual(replaced[0]["revision"], 2)
        self.assertEqual(replaced[0]["supersedes"], f"{previous['experience_id']}@1")


if __name__ == "__main__":
    unittest.main()
