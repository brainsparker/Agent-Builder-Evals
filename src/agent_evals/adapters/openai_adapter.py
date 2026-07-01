from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from agent_evals.models import NormMessage, Task, Trace, Usage
from agent_evals.tools.base import ToolContext, ToolRegistry, ToolSpec


class OpenAIAdapter:
    provider = "openai"

    DEFAULT_INSTRUCTIONS = "Complete the benchmark task. Use tools when needed and cite fetched sources."

    def __init__(
        self,
        model: str = "gpt-5.1",
        api_key: str | None = None,
        runner: Any = None,
        system_prompt: str | None = None,
    ):
        self.model = model
        self.runner = runner
        self.api_key = api_key
        self.system_prompt = system_prompt

    def _to_sdk_tool(self, spec: ToolSpec, registry: ToolRegistry, ctx: ToolContext) -> Any:
        from agents import FunctionTool

        async def invoke(_run_ctx: Any, args_json: str) -> str:
            args = json.loads(args_json or "{}")
            return registry.run(spec.name, args, ctx)

        return FunctionTool(
            name=spec.name,
            description=spec.description,
            params_json_schema=spec.input_schema,
            on_invoke_tool=invoke,
        )

    def _usage(self, result: Any) -> Usage:
        usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
        if usage is None:
            raw = getattr(result, "raw_responses", []) or []
            input_tokens = sum(int(getattr(getattr(item, "usage", None), "input_tokens", 0) or 0) for item in raw)
            output_tokens = sum(int(getattr(getattr(item, "usage", None), "output_tokens", 0) or 0) for item in raw)
            return Usage(input_tokens=input_tokens, output_tokens=output_tokens, llm_calls=len(raw))
        return Usage(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            llm_calls=int(getattr(usage, "requests", 0) or getattr(usage, "llm_calls", 1) or 1),
        )

    def _messages(self, result: Any) -> list[NormMessage]:
        messages = [NormMessage(role="user", content="")]
        final_output = str(getattr(result, "final_output", "") or "")
        if final_output:
            messages.append(NormMessage(role="assistant", content=final_output))
        return messages

    def run_task(self, task: Task, registry: ToolRegistry, ctx: ToolContext) -> Trace:
        start = time.perf_counter()
        error: str | None = None
        finished_reason = "completed"
        final_output = ""
        usage = Usage()
        messages = [NormMessage(role="user", content=task.prompt)]
        allowed = task.allowed_tools or [spec.name for spec in registry.specs()]
        try:
            from agents import Agent, Runner

            sdk_tools = [self._to_sdk_tool(spec, registry, ctx) for spec in registry.specs(allowed)]
            agent = Agent(
                name="Agent Builder Eval Agent",
                model=self.model,
                instructions=self.system_prompt or self.DEFAULT_INSTRUCTIONS,
                tools=sdk_tools,
            )
            runner = self.runner or Runner
            result = runner.run_sync(agent, task.prompt, max_turns=task.max_tool_calls)
            final_output = str(getattr(result, "final_output", "") or "")
            usage = self._usage(result)
            messages = [NormMessage(role="user", content=task.prompt), *self._messages(result)[1:]]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            finished_reason = "error"
        latency = time.perf_counter() - start
        if latency > task.timeout_s and finished_reason == "completed":
            finished_reason = "timeout"
        if len(ctx.tool_calls) >= task.max_tool_calls and not final_output:
            finished_reason = "max_tool_calls"
        return Trace(
            provider="openai",
            model=self.model,
            messages=messages,
            tool_calls=ctx.tool_calls,
            final_output=final_output,
            usage=usage,
            latency_s=latency,
            error=error,
            finished_reason=finished_reason,  # type: ignore[arg-type]
        )
