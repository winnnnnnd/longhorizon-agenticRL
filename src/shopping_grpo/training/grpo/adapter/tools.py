"""veRL 原生工具适配：复用本项目唯一的 ShopSimulator tool schema 与动作守卫。"""

from __future__ import annotations

import asyncio
import math
from typing import Any
from uuid import uuid4

from shopping_grpo.environment.actions import action_reject_reason
from shopping_grpo.environment.client import is_explicit_external_error
from shopping_grpo.environment.evidence import (
    classify_timeout,
    record_empty_tool_evidence,
    record_tool_evidence,
)
from shopping_grpo.environment.tools import tool_call_to_action
from shopping_grpo.environment.observation import render_structured_observation
from shopping_grpo.training.grpo.adapter.runtime import (
    current_environment,
    current_runtime_state,
    record_action_attempt,
    validate_reward,
)

try:  # 本地单测不安装 veRL；部署时由 veRL 注入真实类型。
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import ToolResponse
    from verl.utils.rollout_trace import rollout_trace_op
except ImportError:  # pragma: no cover - 仅轻量开发环境使用
    class ToolResponse:
        def __init__(self, text=None, image=None, video=None):
            self.text, self.image, self.video = text, image, video

    class BaseTool:
        def __init__(self, config, tool_schema):
            self.config, self.tool_schema = config, tool_schema
            function = tool_schema.get("function", {}) if isinstance(tool_schema, dict) else tool_schema.function
            self.name = function.get("name") if isinstance(function, dict) else function.name

    def rollout_trace_op(function):
        return function


class ShopSimulatorTool(BaseTool):
    """执行共享工具定义；当前 coroutine 的 env/state 由 AgentLoop 绑定。"""

    async def create(self, instance_id=None, **kwargs):
        del kwargs
        return instance_id or str(uuid4()), ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        """校验、执行一次工具调用，并把公共结果写入 trajectory 状态。"""
        del instance_id, kwargs
        env = current_environment.get()
        state = current_runtime_state.get()
        if env is None or state is None:
            raise RuntimeError("ShopSimulator tool executed without a trajectory-local interaction state")
        if state["done"] or state["terminate"]:
            return ToolResponse(text="Error: environment is already terminal; do not call another tool."), 0.0, {}
        if len(state["steps"]) >= state["max_steps"]:
            _terminate_max_steps(state)
            return ToolResponse(text="Error: maximum executed tool steps reached."), 0.0, {"reason": "max_steps"}
        parameters = parameters if isinstance(parameters, dict) else {}
        # think 不触碰环境，只记录一次模型决策；其余工具必须经过动作守卫。
        if self.name == "think":
            step = _append_step(state, self.name, parameters)
            _attach_evidence(
                state,
                step,
                tool_name=self.name,
                parameters=parameters,
                observation_state=None,
            )
            if len(state["steps"]) >= state["max_steps"]:
                _terminate_max_steps(state)
                return ToolResponse(text="Error: maximum executed tool steps reached."), 0.0, step
            return ToolResponse(text="Reasoning recorded. Continue with one environment tool call."), 0.0, step
        observation = state.get("latest_observation", "")
        record_action_attempt(state, self.name, parameters, observation)
        state["action_attempt_after_truncation_count"] += int(
            bool(state.get("latest_observation_truncated"))
        )
        reason = action_reject_reason(self.name, parameters, observation)
        if reason:
            _record_nonexecuted_attempt(
                state,
                tool_name=self.name,
                parameters=parameters,
                outcome="guard_rejection",
            )
            state["guard_rejection_count"] += 1
            reason_counts = state["guard_rejection_reason_counts"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            state["guard_rejection_after_truncation_count"] += int(
                bool(state.get("latest_observation_truncated"))
            )
            state["consecutive_guard_rejections"] += 1
            if state["consecutive_guard_rejections"] >= 3:
                _terminate(state, "too_many_guard_rejections")
                return ToolResponse(text="Error: maximum consecutive action guard rejections reached."), 0.0, {
                    "reason": reason
                }
            return ToolResponse(text=f"Error: action guard rejected this call ({reason}); read the latest observation."), 0.0, {"reason": reason}
        try:
            # 先转换成环境动作，再在线程中调用同步客户端；终局 reward 只信任
            # 环境返回的 Reward v3 结构，避免训练侧自行猜测分数。
            action = tool_call_to_action(self.name, parameters)
            result = await asyncio.to_thread(env.step, action)
            if result.get("observation_state") is not None:
                observation = render_structured_observation(
                    result["observation_state"]
                )
            else:
                observation = str(
                    result.get("instruction", result.get("observation", ""))
                )
            step = _append_step(
                state,
                self.name,
                parameters,
                done=bool(result.get("done", False)),
                reward=float(result.get("reward", 0.0)),
            )
        except Exception as exc:
            external_error = is_explicit_external_error(exc)
            _record_nonexecuted_attempt(
                state,
                tool_name=self.name,
                parameters=parameters,
                outcome="external_error" if external_error else "tool_error",
            )
            _terminate(
                state,
                f"tool_error:{exc.__class__.__name__}:{exc}",
                infrastructure_invalid=True,
                external_error=external_error,
            )
            return ToolResponse(text=f"Error: ShopSimulator tool execution failed: {exc}"), 0.0, {"error": state["error"]}
        try:
            _attach_evidence(
                state,
                step,
                tool_name=self.name,
                parameters=parameters,
                observation_state=result.get("observation_state"),
                environment_action=action,
            )
        except Exception as exc:
            _terminate(
                state,
                f"evidence_store_error:{exc.__class__.__name__}:{exc}",
                infrastructure_invalid=True,
            )
            step["evidence_error"] = state["error"]
            return ToolResponse(
                text="Error: trajectory evidence update failed; sample is invalid."
            ), 0.0, step
        state["consecutive_guard_rejections"] = 0
        if step["done"]:
            state["done"] = True
            state["terminate"] = True
            state["termination_reason"] = str(
                result.get("termination_reason") or "environment_done"
            )
            state["terminal_result"] = {
                "done": True,
                "over": result.get("over") is True,
            }
            state["final_reward"] = step["reward"]
            if result.get("over") is not True or not math.isfinite(step["reward"]):
                _mark_infrastructure_invalid(state, "invalid_terminal_result")
            else:
                reward_detail = result.get("reward_detail")
                if (
                    isinstance(reward_detail, dict)
                    and reward_detail.get("reward_version")
                    == "shopsimulator-reward-v3"
                ):
                    try:
                        public_detail = validate_reward(reward_detail)
                        if (
                            public_detail.get("terminal_utility", step["reward"])
                            != step["reward"]
                        ):
                            raise ValueError(
                                "terminal_utility differs from terminal reward"
                            )
                    except ValueError as exc:
                        _mark_infrastructure_invalid(
                            state,
                            f"invalid_terminal_reward_detail:{exc}",
                        )
                    else:
                        state["reward_version"] = public_detail["reward_version"]
                        state["reward_type"] = public_detail["reward_type"]
                        state["reward_valid"] = public_detail["reward_valid"]
                        state["reward_unverifiable"] = not public_detail["reward_valid"]
                        state["reward_detail"] = public_detail
                        state["termination_reason"] = public_detail[
                            "termination_reason"
                        ]
                        if public_detail["reward_type"] == "max_steps":
                            state["timeout_type"] = classify_timeout(
                                state["evidence_store"]
                            )
                            state["outcome_classification"] = state[
                                "timeout_type"
                            ]
                else:
                    _mark_infrastructure_invalid(
                        state,
                        "invalid_terminal_reward_detail:expected Reward v3",
                    )
            return ToolResponse(text="Environment terminated."), 0.0, step
        state["latest_observation"] = observation
        state["latest_observation_raw"] = observation
        state["_pending_raw_observation"] = observation
        if len(state["steps"]) >= state["max_steps"]:
            _terminate_max_steps(state)
            return ToolResponse(text="Error: maximum executed tool steps reached."), 0.0, step
        return ToolResponse(text=observation), 0.0, step

    async def release(self, instance_id, **kwargs):
        # veRL 0.8 会在每次 tool call 后执行 release；真正的环境租约由 AgentLoop 释放。
        del instance_id, kwargs


def _append_step(state, tool, parameters, done=False, reward=0.0):
    """追加一个可审计的工具步骤，并返回刚写入的记录。"""
    step = {
        "index": len(state["steps"]),
        "tool": tool,
        "parameters": parameters,
        "done": bool(done),
        "reward": float(reward),
    }
    state["steps"].append(step)
    return step


def _attach_evidence(
    state,
    step,
    *,
    tool_name,
    parameters,
    observation_state,
    environment_action=None,
):
    """Attach the trajectory Evidence Store's audit fields to one step."""

    if isinstance(observation_state, dict):
        evidence_step = record_tool_evidence(
            state["evidence_store"],
            tool_name=tool_name,
            arguments=parameters,
            observation_state=observation_state,
            step_index=len(state["evidence_store"]["steps"]),
            environment_action=environment_action,
        )
    else:
        evidence_step = record_empty_tool_evidence(
            state["evidence_store"],
            tool_name=tool_name,
            arguments=parameters,
            step_index=len(state["evidence_store"]["steps"]),
        )
    for key in (
        "action_hash",
        "result_hash",
        "result_product_ids",
        "evidence_delta",
        "has_progress",
        "repeat_type",
        "consecutive_no_progress_steps",
        "constraint_coverage",
    ):
        step[key] = evidence_step[key]
    if state["action_attempt_log"]:
        latest_attempt = state["action_attempt_log"][-1]
        if latest_attempt.get("action_hash") == evidence_step["action_hash"]:
            latest_attempt.update(
                {
                    "outcome": "executed",
                    "result_hash": evidence_step["result_hash"],
                    "evidence_delta": evidence_step["evidence_delta"],
                    "has_progress": evidence_step["has_progress"],
                    "repeat_type": evidence_step["repeat_type"],
                    "consecutive_no_progress_steps": evidence_step[
                        "consecutive_no_progress_steps"
                    ],
                    "constraint_coverage": evidence_step[
                        "constraint_coverage"
                    ],
                }
            )
    state["repeat_action_count"] = _repeat_environment_action_count(state)


def _record_nonexecuted_attempt(
    state,
    *,
    tool_name,
    parameters,
    outcome,
):
    """Audit a Guard/tool error result without fabricating environment facts."""

    evidence_step = record_empty_tool_evidence(
        state["evidence_store"],
        tool_name=tool_name,
        arguments=parameters,
        step_index=len(state["evidence_store"]["steps"]),
    )
    if state["action_attempt_log"]:
        state["action_attempt_log"][-1].update(
            {
                "outcome": str(outcome),
                "result_hash": evidence_step["result_hash"],
                "evidence_delta": evidence_step["evidence_delta"],
                "has_progress": evidence_step["has_progress"],
                "repeat_type": evidence_step["repeat_type"],
                "consecutive_no_progress_steps": evidence_step[
                    "consecutive_no_progress_steps"
                ],
                "constraint_coverage": evidence_step[
                    "constraint_coverage"
                ],
            }
        )
    state["repeat_action_count"] = _repeat_environment_action_count(state)


def _repeat_environment_action_count(state):
    return sum(
        1
        for item in state["evidence_store"]["steps"]
        if item.get("tool") != "think" and item.get("repeat_type") is not None
    )


def _mark_infrastructure_invalid(state, reason):
    state["infrastructure_invalid"] = True
    state["termination_reason"] = reason
    state["error"] = reason


def _terminate(
    state,
    reason,
    *,
    infrastructure_invalid=False,
    external_error=False,
):
    """标记 trajectory 停止；基础设施错误不会被误当成模型奖励。"""
    state["terminate"] = True
    state["termination_reason"] = reason
    state["error"] = reason
    if infrastructure_invalid:
        state["infrastructure_invalid"] = True
    if external_error:
        state["external_error"] = True
        state["outcome_classification"] = "external_error"


def _terminate_max_steps(state):
    state["timeout_type"] = classify_timeout(state["evidence_store"])
    state["outcome_classification"] = state["timeout_type"]
    _terminate(state, "max_steps")
