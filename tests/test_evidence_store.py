"""Trajectory Evidence Store contracts and progress semantics."""

import asyncio
import json
import unittest

from shopping_grpo.environment.client import (
    ShopHttpError,
    is_explicit_external_error,
)
from shopping_grpo.environment.evidence import (
    action_hash,
    classify_timeout,
    create_evidence_store,
    record_tool_evidence,
)
from shopping_grpo.training.grpo.adapter.runtime import (
    current_environment,
    current_runtime_state,
    make_runtime_state,
    reward_breakdown,
)
from shopping_grpo.training.grpo.adapter.tools import ShopSimulatorTool
from shopping_grpo.training.grpo.adapter.session import ShopSimulatorSession
from shopping_grpo.training.grpo.dynamic_sampling import aggregate_shopping_metrics
from shopping_grpo.evaluation.rollout import collect_for_task


def contract(*, category="鞋靴", core=None, option=None):
    rows = [
        {
            "constraint_id": "category",
            "constraint_type": "category",
            "attribute": "product.category",
            "operator": "in_category",
            "expected_value": category,
        }
    ]
    if core:
        rows.append(
            {
                "constraint_id": "core",
                "constraint_type": "core_function",
                "attribute": "product.key_attributes",
                "operator": "contains",
                "expected_value": core,
            }
        )
    if option:
        rows.append(
            {
                "constraint_id": "size",
                "constraint_type": "option",
                "attribute": "product.available_options.size",
                "operator": "eq",
                "expected_value": {"value": option, "source_axis": "尺码"},
            }
        )
    return {"version": "shopping-evidence-constraints-v1", "constraints": rows}


def search_state(asins, *, category="鞋靴"):
    return {
        "observation_version": "shopping-observation-v2",
        "page_type": "search_results",
        "products": [
            {
                "asin": asin,
                "title": f"跑步鞋 {asin}",
                "brand": "测试品牌",
                "category": category,
                "price": 399,
                "key_attributes": [],
            }
            for asin in asins
        ],
    }


def detail_state(
    asin="12345678",
    *,
    category="鞋靴",
    attributes=None,
    options=None,
    stock=None,
):
    product = {
        "asin": asin,
        "title": "防水跑步鞋",
        "brand": "测试品牌",
        "category": category,
        "price": 399,
        "key_attributes": attributes or ["防水"],
    }
    if stock is not None:
        product["stock"] = stock
    return {
        "observation_version": "shopping-observation-v2",
        "page_type": "product_detail",
        "product": product,
        "selected_options": {},
        "available_options": options or {"尺码": ["41", "42"]},
    }


def record(store, tool, arguments, state, index):
    return record_tool_evidence(
        store,
        tool_name=tool,
        arguments=arguments,
        observation_state=state,
        step_index=index,
    )


def adapter_tool(name):
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": "test tool",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    return ShopSimulatorTool({}, schema)


class EvidenceStoreTest(unittest.TestCase):
    def test_action_hash_ignores_parameter_order_case_space_and_empty_defaults(self):
        first = action_hash(
            " SEARCH_PRODUCTS ",
            {"query": " Red   Shoes ", "optional": None},
        )
        second = action_hash("search_products", {"query": "red shoes"})
        self.assertEqual(first, second)

    def test_repeated_search_same_result_has_zero_delta(self):
        store = create_evidence_store(
            task_id=1,
            requirement_text="买跑步鞋",
            constraint_contract=contract(),
        )
        state = search_state(["12345678", "12345679"])
        first = record(store, "search_products", {"query": "跑步鞋"}, state, 0)
        second = record(store, "search_products", {"query": "跑步鞋"}, state, 1)

        self.assertTrue(first["has_progress"])
        self.assertFalse(second["has_progress"])
        self.assertEqual(second["repeat_type"], "no_progress_repeat")
        self.assertEqual(len(store["candidates"]["12345678"]["sources"]), 2)

    def test_rewritten_query_with_overlapping_results_is_semantic_repeat(self):
        store = create_evidence_store(
            task_id=1,
            requirement_text="买跑步鞋",
            constraint_contract=contract(),
        )
        state = search_state(["12345678", "12345679", "12345680"])
        record(store, "search_products", {"query": "防水 跑步鞋"}, state, 0)
        repeated = record(
            store,
            "search_products",
            {"query": "跑鞋 防水"},
            state,
            1,
        )

        self.assertFalse(repeated["has_progress"])
        self.assertEqual(repeated["repeat_type"], "semantic_repeat")

    def test_opening_new_detail_adds_relevant_attribute_facts(self):
        store = create_evidence_store(
            task_id=1,
            requirement_text="买防水跑步鞋",
            constraint_contract=contract(core="防水"),
        )
        record(
            store,
            "search_products",
            {"query": "防水跑步鞋"},
            search_state(["12345678"]),
            0,
        )
        detail = record(
            store,
            "open_product",
            {"asin": "12345678"},
            detail_state(),
            1,
        )

        self.assertTrue(detail["evidence_delta"]["new_facts"])
        self.assertIn("core", detail["evidence_delta"]["newly_verified_constraints"])

    def test_missing_required_size_is_new_negative_evidence(self):
        store = create_evidence_store(
            task_id=1,
            requirement_text="买42码跑步鞋",
            constraint_contract=contract(option="42"),
        )
        result = record(
            store,
            "open_product",
            {"asin": "12345678"},
            detail_state(options={"尺码": ["40", "41"]}),
            0,
        )

        self.assertTrue(result["has_progress"])
        self.assertIn("size", result["evidence_delta"]["newly_failed_constraints"])
        negative = [
            fact
            for fact in store["facts"].values()
            if fact["attribute"].startswith("required_option_available:")
        ]
        self.assertEqual(len(negative), 1)
        self.assertFalse(negative[0]["value"])

    def test_repeated_detail_has_no_delta(self):
        store = create_evidence_store(
            task_id=1,
            requirement_text="买防水跑步鞋",
            constraint_contract=contract(core="防水"),
        )
        state = detail_state()
        record(store, "open_product", {"asin": "12345678"}, state, 0)
        repeated = record(store, "open_product", {"asin": "12345678"}, state, 1)

        self.assertFalse(repeated["has_progress"])
        self.assertEqual(repeated["evidence_delta"]["new_facts"], [])
        self.assertEqual(repeated["repeat_type"], "no_progress_repeat")
        price_facts = [
            fact
            for fact in store["facts"].values()
            if fact["attribute"] == "price"
        ]
        self.assertEqual(len(price_facts), 1)
        self.assertEqual(len(price_facts[0]["sources"]), 2)

    def test_conflicting_constraint_observation_can_be_resolved(self):
        budget_contract = contract()
        budget_contract["constraints"].append(
            {
                "constraint_id": "budget",
                "constraint_type": "budget_upper",
                "attribute": "product.price",
                "operator": "lte",
                "expected_value": {"value": 500, "currency": "CNY"},
            }
        )
        store = create_evidence_store(
            task_id=1,
            requirement_text="买500元以内的跑步鞋",
            constraint_contract=budget_contract,
        )
        affordable = detail_state()
        expensive = detail_state()
        expensive["product"]["price"] = 600

        record(store, "open_product", {"asin": "12345678"}, affordable, 0)
        record(store, "open_product", {"asin": "12345678"}, expensive, 1)
        self.assertEqual(store["constraints"]["budget"]["status"], "conflict")
        resolved = record(
            store,
            "open_product",
            {"asin": "12345678"},
            expensive,
            2,
        )

        self.assertEqual(
            store["constraints"]["budget"]["status"],
            "verified_fail",
        )
        self.assertIn("budget", resolved["evidence_delta"]["resolved_conflicts"])

    def test_new_irrelevant_products_do_not_count_as_progress(self):
        store = create_evidence_store(
            task_id=1,
            requirement_text="买跑步鞋",
            constraint_contract=contract(),
        )
        result = record(
            store,
            "search_products",
            {"query": "厨房电器"},
            search_state(["87654321"], category="厨房电器"),
            0,
        )

        self.assertFalse(result["has_progress"])
        self.assertEqual(result["evidence_delta"]["new_relevant_candidates"], [])
        self.assertEqual(store["relevant_product_ids"], [])

    def test_changed_stock_is_progress_not_a_repeat(self):
        store = create_evidence_store(
            task_id=1,
            requirement_text="买跑步鞋",
            constraint_contract=contract(),
        )
        first = detail_state(stock=3)
        second = detail_state(stock=2)
        record(store, "open_product", {"asin": "12345678"}, first, 0)
        changed = record(store, "open_product", {"asin": "12345678"}, second, 1)

        self.assertTrue(changed["has_progress"])
        self.assertIsNone(changed["repeat_type"])

    def test_external_error_and_model_no_progress_are_separate(self):
        self.assertTrue(is_explicit_external_error(ShopHttpError("HTTP 500")))
        self.assertTrue(is_explicit_external_error(TimeoutError("timed out")))
        self.assertFalse(is_explicit_external_error(ValueError("bad model args")))

        store = create_evidence_store(
            task_id=1,
            requirement_text="买跑步鞋",
            constraint_contract=contract(),
        )
        irrelevant = search_state(["87654321"], category="厨房电器")
        for index in range(4):
            record(
                store,
                "search_products",
                {"query": f"无关查询 {index}"},
                irrelevant,
                index,
            )
        self.assertEqual(classify_timeout(store), "no_progress_timeout")
        self.assertNotIn("external_error", classify_timeout(store))

        productive = create_evidence_store(
            task_id=2,
            requirement_text="买跑步鞋",
            constraint_contract=contract(),
        )
        record(
            productive,
            "search_products",
            {"query": "跑步鞋"},
            search_state(["12345678"]),
            0,
        )
        self.assertEqual(classify_timeout(productive), "productive_timeout")

    def test_adapter_marks_only_explicit_transport_failure_as_external(self):
        class FailingEnvironment:
            def step(self, action):
                del action
                raise ShopHttpError("HTTP 500")

        async def run():
            state = make_runtime_state(
                task_id=1,
                max_steps=35,
                requirement_text="买跑步鞋",
                constraint_contract=contract(),
            )
            state["latest_observation"] = "搜索功能是否可用: True"
            environment_token = current_environment.set(FailingEnvironment())
            state_token = current_runtime_state.set(state)
            try:
                await adapter_tool("search_products").execute(
                    "call-1",
                    {"query": "跑步鞋"},
                )
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(environment_token)
            return state

        state = asyncio.run(run())

        self.assertTrue(state["external_error"])
        self.assertEqual(state["outcome_classification"], "external_error")
        self.assertEqual(
            state["action_attempt_log"][0]["outcome"],
            "external_error",
        )

    def test_store_is_json_serializable_for_ray_logs_and_replay(self):
        store = create_evidence_store(
            task_id=1,
            requirement_text="买42码跑步鞋",
            constraint_contract=contract(option="42"),
        )
        record(
            store,
            "open_product",
            {"asin": "12345678"},
            detail_state(),
            0,
        )
        encoded = json.dumps(store, ensure_ascii=False, sort_keys=True)
        self.assertEqual(json.loads(encoded)["version"], "shopping-evidence-store-v1")

    def test_store_lifecycle_is_bound_to_one_trajectory_session(self):
        instances = []

        class Environment:
            def __init__(self, **kwargs):
                del kwargs
                self.released = False
                instances.append(self)

            def reset(self, task_id):
                return {
                    "instruction": f"task {task_id}",
                    "environment_version": "shopsimulator-environment-v2.1",
                    "evidence_constraint_contract": contract(),
                    "observation_state": {
                        "observation_version": "shopping-observation-v2",
                        "page_type": "search_home",
                        "search_available": True,
                        "actions": [],
                    },
                }

            def release(self):
                self.released = True

        async def run():
            session = ShopSimulatorSession(
                required_environment_version="shopsimulator-environment-v2.1",
                env_factory=Environment,
            )
            state = await session.start(7)
            self.assertIs(current_runtime_state.get(), state)
            await session.close()
            self.assertIsNone(current_runtime_state.get())
            return state

        state = asyncio.run(run())

        self.assertEqual(state["evidence_store"]["task_id"], 7)
        self.assertTrue(instances[0].released)

    def test_reward_evaluator_exports_evidence_signals_without_changing_utility(self):
        state = make_runtime_state(
            task_id=1,
            max_steps=35,
            requirement_text="买跑步鞋",
            constraint_contract=contract(),
        )
        record(
            state["evidence_store"],
            "search_products",
            {"query": "跑步鞋"},
            search_state(["12345678"]),
            0,
        )
        state.update(
            {
                "done": True,
                "terminal_result": {"done": True, "over": True},
                "final_reward": 1.0,
                "reward_version": "shopsimulator-reward-v3",
                "reward_type": "gold_purchase",
                "reward_valid": True,
                "reward_detail": {
                    "weighted_score": 1.0,
                    "evidence_coverage": 1.0,
                    "dimension_scores": {},
                    "hard_gates": {},
                },
            }
        )

        evaluated = reward_breakdown(state)

        self.assertEqual(evaluated["total"], 1.0)
        self.assertEqual(evaluated["evidence_progress_rate"], 1.0)
        self.assertEqual(evaluated["evidence_progress_steps"], 1)
        metrics = aggregate_shopping_metrics(
            [
                {
                    "steps": 1,
                    "done": True,
                    "termination_reason": "gold_purchase",
                    "reward_type": "gold_purchase",
                    "reward": evaluated,
                }
            ]
        )
        self.assertEqual(metrics["trajectory/evidence_progress_rate"], 1.0)
        self.assertEqual(metrics["trajectory/no_progress_repeat_mean"], 0.0)

    def test_evaluation_rollout_logs_store_without_injecting_it_into_messages(self):
        class Client:
            def __init__(self):
                self.requests = []
                self.calls = [
                    ("search_products", {"query": "跑步鞋"}),
                    ("open_product", {"asin": "12345678"}),
                    ("buy_now", {}),
                ]

            def complete(self, messages, tools):
                del tools
                self.requests.append(json.dumps(messages, ensure_ascii=False))
                name, arguments = self.calls.pop(0)
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{len(self.requests)}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }

        class Environment:
            def __init__(self, **kwargs):
                del kwargs
                self.released = False

            def reset(self, task_id):
                return {
                    "env_idx": 1,
                    "instruction": "买一双跑步鞋",
                    "evidence_constraint_contract": contract(),
                    "observation_state": {
                        "observation_version": "shopping-observation-v2",
                        "page_type": "search_home",
                        "search_available": True,
                        "actions": [],
                    },
                }

            def step(self, action):
                if action.startswith("search["):
                    state = search_state(["12345678"])
                    state.update(
                        {
                            "search_available": False,
                            "actions": ["back to search", "12345678"],
                            "query": "跑步鞋",
                            "normalized_query": "跑步鞋",
                            "page": 1,
                            "total_pages": 1,
                            "total_results": 1,
                            "rank_start": 1,
                            "rank_end": 1,
                        }
                    )
                    state["products"][0]["rank"] = 1
                    return {"observation_state": state, "reward": 0.0, "done": False}
                if action == "click[12345678]":
                    state = detail_state()
                    state.update(
                        {
                            "search_available": False,
                            "actions": ["back to search", "buy now"],
                        }
                    )
                    return {"observation_state": state, "reward": 0.0, "done": False}
                return {
                    "observation_state": {
                        "observation_version": "shopping-observation-v2",
                        "page_type": "terminal",
                        "search_available": False,
                        "actions": [],
                    },
                    "reward": 1.0,
                    "done": True,
                    "over": True,
                    "reward_detail": {"reward_type": "gold_purchase"},
                }

            def release(self):
                self.released = True

        client = Client()
        environment = Environment()
        trajectory = collect_for_task(
            {"task_id": 1},
            client=client,
            env_factory=lambda **kwargs: environment,
            max_steps=4,
        )

        self.assertEqual(trajectory["status"], "done")
        self.assertTrue(environment.released)
        self.assertTrue(trajectory["steps"][1]["has_progress"])
        self.assertEqual(
            trajectory["evidence_store"]["version"],
            "shopping-evidence-store-v1",
        )
        self.assertTrue(
            all("shopping-evidence-store-v1" not in request for request in client.requests)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
