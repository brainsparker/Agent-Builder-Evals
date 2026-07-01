from __future__ import annotations

from typing import Protocol

from agent_evals.models import Task, Trace
from agent_evals.tools.base import ToolContext, ToolRegistry


class AgentAdapter(Protocol):
    provider: str
    model: str

    def run_task(self, task: Task, registry: ToolRegistry, ctx: ToolContext) -> Trace:
        ...


def make_adapter(
    provider: str,
    model: str,
    api_key: str | None = None,
    system_prompt: str | None = None,
) -> AgentAdapter:
    if provider == "anthropic":
        from agent_evals.adapters.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(model=model, api_key=api_key, system_prompt=system_prompt)
    if provider == "openai":
        from agent_evals.adapters.openai_adapter import OpenAIAdapter

        return OpenAIAdapter(model=model, api_key=api_key, system_prompt=system_prompt)
    raise ValueError(f"Unsupported provider: {provider}")
