from __future__ import annotations

from datetime import UTC, datetime
import time
from typing import Any

from agent_evals.models import NormMessage, Task, Trace, Usage
from agent_evals.tools.base import ToolContext, ToolRegistry, ToolSpec


class AnthropicAdapter:
    provider = "anthropic"

    def __init__(self, model: str = "claude-opus-4-8", api_key: str | None = None, client: Any = None):
        self.model = model
        if client is not None:
            self.client = client
        else:
            import anthropic

            self.client = anthropic.Anthropic(api_key=api_key)

    def _tool_defs(self, specs: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
                "strict": True,
            }
            for spec in specs
        ]

    def _usage(self, response: Any) -> Usage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return Usage(llm_calls=1)
        return Usage(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_creation_input_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            llm_calls=1,
        )

    def _content_text(self, content: Any) -> str:
        parts: list[str] = []
        for block in content or []:
            if getattr(block, "type", None) == "text":
                parts.append(str(getattr(block, "text", "")))
        return "\n".join(parts).strip()

    def _tool_uses(self, content: Any) -> list[Any]:
        return [block for block in content or [] if getattr(block, "type", None) == "tool_use"]

    def run_task(self, task: Task, registry: ToolRegistry, ctx: ToolContext) -> Trace:
        start = time.perf_counter()
        messages: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
        norm_messages = [NormMessage(role="user", content=task.prompt)]
        usage = Usage()
        final_output = ""
        error: str | None = None
        finished_reason = "completed"
        allowed = task.allowed_tools or [spec.name for spec in registry.specs()]
        tool_defs = self._tool_defs(registry.specs(allowed))
        try:
            for _ in range(task.max_tool_calls + 1):
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    tools=tool_defs,
                    messages=messages,
                )
                usage = usage.add(self._usage(response))
                norm_messages.append(NormMessage(role="assistant", content=str(response.content)))
                final_output = self._content_text(response.content)
                tool_uses = self._tool_uses(response.content)
                if not tool_uses:
                    break
                if len(ctx.tool_calls) + len(tool_uses) > task.max_tool_calls:
                    finished_reason = "max_tool_calls"
                    break
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for tool_use in tool_uses:
                    result = registry.run(
                        str(tool_use.name),
                        dict(getattr(tool_use, "input", {}) or {}),
                        ctx,
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": result,
                        }
                    )
                messages.append({"role": "user", "content": tool_results})
                norm_messages.append(NormMessage(role="user", content=tool_results))
            else:
                finished_reason = "max_tool_calls"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            finished_reason = "error"
        latency = time.perf_counter() - start
        if latency > task.timeout_s and finished_reason == "completed":
            finished_reason = "timeout"
        return Trace(
            provider="anthropic",
            model=self.model,
            messages=norm_messages,
            tool_calls=ctx.tool_calls,
            final_output=final_output,
            usage=usage,
            latency_s=latency,
            error=error,
            finished_reason=finished_reason,  # type: ignore[arg-type]
        )
