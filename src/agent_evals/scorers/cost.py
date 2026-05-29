from __future__ import annotations

from agent_evals.models import Task, Trace
from agent_evals.pricing import cost_for
from agent_evals.scorers.base import ScoreResult, clamp


COST_BUDGET_USD = 0.25


class CostScorer:
    name = "cost"

    def score(self, task: Task, trace: Trace) -> ScoreResult:
        cost, warning = cost_for(trace.model, trace.usage)
        if cost is None:
            return ScoreResult(self.name, 1, raw={"warning": warning}, reason=warning or "", applicable=False)
        return ScoreResult(self.name, clamp(1 - cost / COST_BUDGET_USD), raw=cost)
