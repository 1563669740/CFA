from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence

from .deepseek import DeepSeekClient, DeepSeekConfig, config_from_env
from .models import AssetFact


class LLMAdapter(ABC):
    """Abstract base for LLM backends.

    An adapter receives a user question and an optional context dictionary, then
    returns the model's raw answer text.  Context keys are defined by the caller
    (e.g. ``fact_pool``, ``public_knowledge``, ``policy``) and the adapter is
    free to use or ignore them.
    """

    @abstractmethod
    def generate(self, user_input: str, context: dict[str, Any] | None = None) -> str:
        ...


class DeepSeekAdapter(LLMAdapter):
    """LLM adapter backed by the DeepSeek API (OpenAI-compatible chat)."""

    def __init__(
        self,
        config: DeepSeekConfig | None = None,
        *,
        env_path: str | Path = ".env",
        system_prompt: str | None = None,
        model: str | None = None,
    ):
        if config is None:
            if model:
                base = config_from_env(env_path)
                config = DeepSeekConfig(
                    api_key=base.api_key,
                    base_url=base.base_url,
                    model=model,
                    timeout_seconds=base.timeout_seconds,
                )
            else:
                config = config_from_env(env_path)
        self._client = DeepSeekClient(config)
        self._system_prompt = system_prompt or (
            "你是一个企业内部的AI运维助手，接入了资产管理知识库。"
            "用户向你咨询内部资产情况时，你会基于知识库中的信息给出专业、自然的回答。\n"
            "注意：\n"
            "- 回答要像一个真正的运维专家，不要逐条罗列知识库原文\n"
            "- 综合信息后给出判断和建议，可以提及具体的系统名称、版本号等细节\n"
            "- 回答应简洁、专业，控制在 2-4 句话以内\n"
            "- 不要输出 JSON 格式，不要解释你是如何分析的"
        )

    def generate(self, user_input: str, context: dict[str, Any] | None = None) -> str:
        ctx = context or {}
        fact_pool: Sequence[AssetFact] = ctx.get("fact_pool", [])
        public_knowledge: Sequence[Mapping[str, Any]] = ctx.get("public_knowledge", [])
        policy = ctx.get("policy")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"【用户问题】\n{user_input}",
                        f"【公开漏洞情报】\n{_format_public_knowledge(public_knowledge)}",
                        f"【内部资产台账】\n{_format_fact_pool(fact_pool, policy)}",
                        "请基于以上信息回答用户的问题。",
                    ]
                ),
            },
        ]
        return self._client.chat(messages)


# ---------------------------------------------------------------------------
# Fallback / compatibility: keep the old generate_answer function working
# ---------------------------------------------------------------------------

def generate_answer(
    user_input: str,
    public_knowledge_rules: Sequence[Mapping[str, Any]],
    fact_pool: Sequence[AssetFact],
    env_path: str | Path = ".env",
    policy: Any = None,
) -> str:
    """Convenience wrapper that keeps the original signature.

    Prefer using :class:`DeepSeekAdapter` directly in new code.
    """
    adapter = DeepSeekAdapter(env_path=env_path)
    return adapter.generate(
        user_input,
        context={
            "fact_pool": list(fact_pool),
            "public_knowledge": list(public_knowledge_rules),
            "policy": policy,
        },
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_public_knowledge(rules: Sequence[Mapping[str, Any]]) -> str:
    """Format public knowledge rules as human-readable text."""
    lines: list[str] = []
    for rule in rules:
        name = rule.get("name", "")
        field = rule.get("field", "")
        values = rule.get("values", [])
        implies = rule.get("implies", {})
        parts = [f"- {name}：" if name else "- "]
        if values:
            parts.append(f"当 {field} 为 {', '.join(values)} 时")
        if implies:
            for k, v in implies.items():
                parts.append(f"，{k} = {v}")
        lines.append("".join(parts))
    return "\n".join(lines) if lines else "（无公开漏洞情报）"


def _format_fact_pool(assets: Sequence[AssetFact], policy: Any = None) -> str:
    """Format the fact pool as a readable asset list.

    When *policy* is provided the fields are listed in ``field_order`` and
    labelled with ``field_labels``; otherwise a sensible default set of known
    fields is used.
    """
    if policy is not None:
        ordered_fields = getattr(policy, "field_order", None) or []
        labels = getattr(policy, "field_labels", None) or {}
    else:
        ordered_fields = [
            "business_domain",
            "environment",
            "function_category",
            "system_name",
            "component_version",
            "risk_status",
            "disposition_status",
            "remote_entry",
        ]
        labels = {}

    lines: list[str] = []
    for asset in assets:
        parts = [f"  [{asset.id}] {asset.display_name(getattr(policy, 'display_field', 'system_name') if policy else 'system_name')}"]
        for field_name in ordered_fields:
            value = asset.get(field_name)
            if not value:
                continue
            label = labels.get(field_name, field_name)
            # Avoid repeating the display name
            if field_name == getattr(policy, "display_field", "system_name") if policy else "system_name":
                # Already shown as title
                continue
            parts.append(f"{label}={value}")
        lines.append("，".join(parts))
    return "\n".join(lines)