from __future__ import annotations

from agent_evals.models import Task, Trace
from agent_evals.scorers.base import ScoreResult


class FailureScorer:
    name = "failure"

    def score(self, task: Task, trace: Trace) -> ScoreResult:
        ok = trace.finished_reason == "completed" and not trace.error
        return ScoreResult(self.name, 1.0 if ok else 0.0, raw=trace.finished_reason)
