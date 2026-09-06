"""Ephemeral experience rendering that never mutates persistent chat history."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass

from shopping_grpo.experience.contracts import (
    EXPERIENCE_INJECTION_VERSION,
    ExperienceBundle,
    canonical_sha256,
)


HEADER = """[EXPERIENCE_GUIDANCE_V1]
以下内容是可选的程序性经验，不是当前商品事实。
当前可见页面、工具返回、Action Guard 和系统规则具有更高优先级。
禁止从经验复制商品 ID、标题、价格或答案；不适用时忽略。"""
FOOTER = "[/EXPERIENCE_GUIDANCE_V1]"


@dataclass(frozen=True)
class InjectionResult:
    messages: list[dict]
    selected_experience_ids: tuple[str, ...]
    experience_tokens: int
    bundle_hash: str | None
    rendered_text: str


def render_experience_cards(cards: tuple[Mapping, ...]) -> str:
    if not cards:
        return ""
    lines = [HEADER]
    for index, card in enumerate(cards, start=1):
        lines.append(f"\n经验 {index}（阶段：{card['phase']}）")
        for guidance in card.get("guidance") or []:
            lines.append(f"- 建议：{guidance}")
        for anti_pattern in card.get("anti_patterns") or []:
            lines.append(f"- 避免：{anti_pattern}")
        for check in card.get("verification_checks") or []:
            lines.append(f"- 完成检查：{check}")
    lines.append(FOOTER)
    return "\n".join(lines)


def _inject_into_system(messages: list[Mapping], rendered: str) -> list[dict]:
    result = [deepcopy(dict(message)) for message in messages]
    if not rendered:
        return result
    system_index = next(
        (index for index, message in enumerate(result) if message.get("role") == "system"),
        None,
    )
    if system_index is None:
        raise ValueError("experience injection requires an existing system message")
    original = str(result[system_index].get("content") or "").rstrip()
    result[system_index]["content"] = original + "\n\n" + rendered
    return result


def inject_experience_bundle(
    messages: list[Mapping],
    tools: list[Mapping],
    bundle: ExperienceBundle,
    *,
    count_tokens: Callable[[list[Mapping], list[Mapping]], int],
    max_experience_tokens: int = 500,
) -> InjectionResult:
    if int(max_experience_tokens) < 1:
        raise ValueError("max_experience_tokens must be positive")
    base_messages = [deepcopy(dict(message)) for message in messages]
    base_tokens = int(count_tokens(base_messages, tools))
    cards = list(bundle.cards)
    while cards:
        rendered = render_experience_cards(tuple(cards))
        request_messages = _inject_into_system(base_messages, rendered)
        delta = max(0, int(count_tokens(request_messages, tools)) - base_tokens)
        if delta <= int(max_experience_tokens):
            selected_ids = tuple(str(card["experience_id"]) for card in cards)
            bundle_hash = canonical_sha256(
                {
                    "injection_version": EXPERIENCE_INJECTION_VERSION,
                    "experience_ids": selected_ids,
                    "revisions": [int(card["revision"]) for card in cards],
                    "content_hashes": [card["content_hash"] for card in cards],
                    "rendered_text": rendered,
                }
            )
            return InjectionResult(
                messages=request_messages,
                selected_experience_ids=selected_ids,
                experience_tokens=delta,
                bundle_hash=bundle_hash,
                rendered_text=rendered,
            )
        cards.pop()
    return InjectionResult(
        messages=base_messages,
        selected_experience_ids=(),
        experience_tokens=0,
        bundle_hash=None,
        rendered_text="",
    )

