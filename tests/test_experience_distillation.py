import unittest

from shopping_grpo.experience.distillation import (
    MultiTrajectoryDistiller,
    build_joint_analysis_groups,
    build_trajectory_records,
)
from shopping_grpo.experience.segmentation import actor_visible_events


def transcript(trajectory_id, task_id, query, tool_name="search_products"):
    return {
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "Instruction: " + query},
            {
                "role": "assistant",
                "content": "先搜索",
                "tool_calls": [
                    {
                        "id": trajectory_id + "-call",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": '{"query":"水彩笔"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": trajectory_id + "-call",
                "name": tool_name,
                "content": "[SHOPPING_OBSERVATION_V2]\npage_type: search_results",
            },
        ],
    }


class FakeDistillationClient:
    model = "deepseek-v4-flash"
    model_revision = "deepseek-v4-flash-test-revision"
    provider_id = "fake-dsv4-provider"

    def complete_json(self, messages):
        payload = __import__("json").loads(messages[1]["content"])
        ids = [row["trajectory_id"] for row in payload["group"]["trajectories"]]
        return {
            "result": {
                "schema_version": "shopping-multi-trajectory-distillation-v1",
                "analysis": {"shared_pattern": "搜索应保留区分度高的条件"},
                "experiences": [
                    {
                        "type": "search_strategy",
                        "phase": "search",
                        "trigger": {
                            "events": ["task_start", "search_stagnation"],
                            "page_types": [],
                            "required_constraint_tags": [],
                            "predicates": [],
                        },
                        "guidance": ["搜索时保留品类和最有区分度的条件。"],
                        "anti_patterns": ["不要机械重复没有新增候选的查询。"],
                        "verification_checks": ["新查询应产生新的候选信息。"],
                        "categories": [],
                        "supporting_trajectory_ids": ids[:2],
                        "contradicting_trajectory_ids": ids[2:3],
                    }
                ],
            },
            "metadata": {"requested_model": self.model},
        }


class ExperienceDistillationTest(unittest.TestCase):
    def test_decision_evidence_drops_reward_and_audit_only_observation(self):
        events = actor_visible_events(
            {
                "schema_version": "shopping-normalized-trajectory-v1",
                "events": [
                    {
                        "event_id": "e0001",
                        "event_type": "tool_step",
                        "assistant_text": "查看候选",
                        "tool_name": "open_product",
                        "parameters": {"asin": "123456789012"},
                        "actor_visible_observation": "page_type: product_detail",
                        "reward": 1.0,
                        "env_action": "hidden action",
                        "audit_only_raw_observation": "hidden raw observation",
                    }
                ],
            }
        )

        self.assertNotIn("reward", events[0])
        self.assertNotIn("env_action", events[0])
        self.assertNotIn("audit_only_raw_observation", events[0])

    def test_same_query_teacher_and_repeated_students_are_analyzed_together(self):
        query = "购买适合儿童使用且可水洗的水彩笔"
        teacher = build_trajectory_records(
            [transcript("teacher-success", 1, query)],
            actor_role="teacher",
            source_kind="curated_teacher_gold",
        )
        students = build_trajectory_records(
            [
                transcript("student-failure-1", 1, query),
                transcript("student-failure-2", 1, query),
            ],
            actor_role="student",
            source_kind="agent_rollout",
        )
        groups = build_joint_analysis_groups(teacher + students, maximum_group_size=8)

        member_ids = {
            row["trajectory_id"] for row in groups[0]["trajectories"]
        }
        self.assertEqual(
            member_ids,
            {"teacher-success", "student-failure-1", "student-failure-2"},
        )
        self.assertTrue(groups[0]["has_teacher_student_pair"])

        cards, audit = MultiTrajectoryDistiller(
            FakeDistillationClient(), created_at="2026-09-06T00:00:00+00:00"
        ).distill_group(groups[0])

        self.assertEqual(cards[0]["status"], "candidate")
        self.assertEqual(cards[0]["evidence"]["source_kind"], "teacher_agent_pair")
        self.assertEqual(audit["provider_metadata"]["requested_model"], "deepseek-v4-flash")


if __name__ == "__main__":
    unittest.main()
