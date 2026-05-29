from __future__ import annotations

from typing import Any

from agent_evals.tools.base import ToolContext, ToolSpec, schema


class WebSearchTool:
    spec = ToolSpec(
        name="web_search",
        description="Search the public web for current information. Returns result snippets and URLs.",
        input_schema=schema(
            {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                },
            },
            ["query"],
        ),
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return ctx.search_backend.search(
            query=str(args["query"]),
            max_results=int(args.get("max_results", 5)),
        )
