from __future__ import annotations

from typing import Any

from agent_evals.tools.base import ToolContext, ToolSpec, schema


class FetchUrlTool:
    spec = ToolSpec(
        name="fetch_url",
        description="Fetch and extract readable text from a URL.",
        input_schema=schema(
            {"url": {"type": "string", "description": "The URL to fetch."}},
            ["url"],
        ),
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return ctx.search_backend.extract(str(args["url"]))
