from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import time
from typing import Any, Protocol

from pydantic import BaseModel, Field

from agent_evals.models import NormToolCall


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolContext:
    search_backend: Any
    record_mode: str = "live"
    cassette: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[NormToolCall] = field(default_factory=list)


class Tool(Protocol):
    spec: ToolSpec

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        ...


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {tool.spec.name: tool for tool in tools}

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool {name}") from exc

    def specs(self, names: list[str] | None = None) -> list[ToolSpec]:
        selected = names or sorted(self._tools)
        return [self.get(name).spec for name in selected]

    def run(self, name: str, args: dict[str, Any], ctx: ToolContext) -> str:
        key = json.dumps({"name": name, "args": args}, sort_keys=True)
        started = datetime.now(UTC).isoformat()
        begin = time.perf_counter()
        is_error = False
        try:
            if ctx.record_mode == "replay" and key in ctx.cassette:
                result = str(ctx.cassette[key])
            else:
                result = self.get(name).run(args, ctx)
                ctx.cassette[key] = result
        except Exception as exc:
            is_error = True
            result = f"ERROR: {type(exc).__name__}: {exc}"
        duration = time.perf_counter() - begin
        ctx.tool_calls.append(
            NormToolCall(
                name=name,
                arguments=args,
                result=result,
                is_error=is_error,
                started_at=started,
                duration_s=duration,
            )
        )
        return result


def schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_default_registry() -> ToolRegistry:
    from agent_evals.tools.code_exec import CodeExecTool
    from agent_evals.tools.fetch_url import FetchUrlTool
    from agent_evals.tools.support_data import SupportLookupTool
    from agent_evals.tools.web_search import WebSearchTool

    return ToolRegistry([WebSearchTool(), FetchUrlTool(), CodeExecTool(), SupportLookupTool()])
