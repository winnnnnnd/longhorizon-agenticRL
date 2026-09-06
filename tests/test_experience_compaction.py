import unittest

from shopping_grpo.experience.compaction import (
    DeepSeekFlashContextCompactor,
    SemanticCompactionCache,
    SemanticCompactionError,
)


def assistant(call_id, query):
    return {
        "role": "assistant",
        "content": f"搜索 {query}",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "search_products",
                    "arguments": '{"query":"' + query + '"}',
                },
            }
        ],
    }


def tool(call_id, text):
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": "search_products",
        "content": text,
    }


class FakeSummaryClient:
    model = "deepseek-v4-flash"
    model_revision = "deepseek-v4-flash-test-revision"
    provider_id = "fake-dsv4-provider"

    def __init__(self, bad_quote=False):
        self.calls = []
        self.bad_quote = bad_quote

    def complete_json(self, messages):
        self.calls.append(messages)
        payload = __import__("json").loads(messages[1]["content"])
        events = payload["source_events"]
        quote = "not in source" if self.bad_quote else "旧候选页面"
        return {
            "result": {
                "schema_version": "shopping-context-summary-v1",
                "source_event_ids": [event["event_id"] for event in events],
                "records": [
                    {
                        "kind": "search",
                        "fact": "旧查询只得到旧候选页面。",
                        "evidence": [{"event_id": events[0]["event_id"], "quote": quote}],
                    }
                ],
            },
            "metadata": {"requested_model": self.model},
        }


class ExperienceCompactionTest(unittest.TestCase):
    def test_preserves_anchor_and_recent_groups_and_caches_grounded_summary(self):
        client = FakeSummaryClient()
        compactor = DeepSeekFlashContextCompactor(
            client,
            preserve_recent_groups=2,
            cache=SemanticCompactionCache(None),
        )
        messages = [
            {"role": "system", "content": "fixed rules\n\n[EXPERIENCE_GUIDANCE_V1]\n经验"},
            {"role": "user", "content": "原始 query"},
            assistant("old", "旧查询"),
            tool("old", "旧候选页面"),
            assistant("middle", "中间查询"),
            tool("middle", "中间候选页面"),
            assistant("latest", "最新查询"),
            tool("latest", "当前候选页面"),
        ]

        compacted, event = compactor.compact(
            messages,
            [],
            count_tokens=lambda candidate, tools: len(candidate),
            max_input_tokens=6,
        )
        second, second_event = compactor.compact(
            messages,
            [],
            count_tokens=lambda candidate, tools: len(candidate),
            max_input_tokens=6,
        )

        self.assertEqual(compacted, second)
        self.assertEqual(compacted[1]["content"], "原始 query")
        self.assertIn("[EXPERIENCE_GUIDANCE_V1]", compacted[0]["content"])
        self.assertNotIn("旧候选页面", str(compacted))
        self.assertIn("中间候选页面", str(compacted))
        self.assertIn("当前候选页面", str(compacted))
        self.assertEqual(event["model"], "deepseek-v4-flash")
        self.assertEqual(
            event["model_revision"], "deepseek-v4-flash-test-revision"
        )
        self.assertEqual(event["provider_id"], "fake-dsv4-provider")
        self.assertEqual(second_event["cache_hits"], 1)
        self.assertEqual(len(client.calls), 1)

    def test_rejects_summary_without_exact_evidence_quote(self):
        compactor = DeepSeekFlashContextCompactor(
            FakeSummaryClient(bad_quote=True), preserve_recent_groups=1
        )
        messages = [
            {"role": "system", "content": "fixed rules"},
            {"role": "user", "content": "query"},
            assistant("old", "旧查询"),
            tool("old", "旧候选页面"),
            assistant("latest", "最新查询"),
            tool("latest", "当前候选页面"),
        ]
        with self.assertRaisesRegex(SemanticCompactionError, "not grounded"):
            compactor.compact(
                messages,
                [],
                count_tokens=lambda candidate, tools: len(candidate),
                max_input_tokens=4,
            )


if __name__ == "__main__":
    unittest.main()
