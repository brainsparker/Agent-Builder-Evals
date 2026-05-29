from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_evals.tools.base import ToolContext, ToolSpec, schema


FIXTURE_PATH = Path(__file__).parents[1] / "tasks" / "_fixtures" / "support" / "store.json"


class SupportLookupTool:
    spec = ToolSpec(
        name="support_lookup",
        description="Look up deterministic customer support data and apply allowed support actions.",
        input_schema=schema(
            {
                "action": {
                    "type": "string",
                    "enum": ["get_order", "get_customer", "search_kb", "issue_refund"],
                },
                "args": {"type": "object", "description": "Action-specific arguments."},
            },
            ["action", "args"],
        ),
    )

    def _store(self) -> dict[str, Any]:
        return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        store = self._store()
        action = args["action"]
        params = args.get("args", {})
        if action == "get_order":
            return json.dumps(store["orders"].get(str(params.get("order_id")), {}), sort_keys=True)
        if action == "get_customer":
            return json.dumps(store["customers"].get(str(params.get("customer_id")), {}), sort_keys=True)
        if action == "search_kb":
            query = str(params.get("query", "")).lower()
            hits = [row for row in store["kb"] if query in row["title"].lower() or query in row["body"].lower()]
            return json.dumps(hits, sort_keys=True)
        if action == "issue_refund":
            order = store["orders"].get(str(params.get("order_id")), {})
            amount = float(params.get("amount", 0))
            allowed = order.get("status") in {"delivered", "delayed"} and amount <= float(order.get("total", 0))
            return json.dumps({"issued": allowed, "order_id": params.get("order_id"), "amount": amount})
        raise ValueError(f"Unsupported support action: {action}")
