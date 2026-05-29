from __future__ import annotations

import json

from agent_evals.models import Task, Trace
from agent_evals.scorers.base import ScoreResult, clamp


class ToolAccuracyScorer:
    name = "tool"

    def score(self, task: Task, trace: Trace) -> ScoreResult:
        if not task.expected_tool_calls:
            return ScoreResult(self.name, 1, applicable=False)
        total = len(task.expected_tool_calls)
        earned = 0.0
        calls_by_name: dict[str, list] = {}
        for call in trace.tool_calls:
            calls_by_name.setdefault(call.name, []).append(call)
        details = []
        for expected in task.expected_tool_calls:
            calls = calls_by_name.get(expected.tool, [])
            count = len(calls)
            ok = count >= expected.min_calls and (expected.max_calls is None or count <= expected.max_calls)
            args_ok = True
            args_blob = json.dumps([call.arguments for call in calls], sort_keys=True).lower()
            for key, value in expected.arg_contains.items():
                args_ok = args_ok and str(key).lower() in args_blob and str(value).lower() in args_blob
            if ok and args_ok:
                earned += 1
            details.append({"tool": expected.tool, "count": count, "ok": ok, "args_ok": args_ok})
        error_penalty = sum(1 for call in trace.tool_calls if call.is_error) * 0.1
        thrash_penalty = max(0, len(trace.tool_calls) - task.max_tool_calls) * 0.1
        return ScoreResult(self.name, clamp(earned / total - error_penalty - thrash_penalty), raw=details)
