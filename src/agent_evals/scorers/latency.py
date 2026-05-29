from __future__ import annotations

from agent_evals.models import Category, Task, Trace
from agent_evals.scorers.base import ScoreResult, clamp


BUDGETS = {
    Category.coding: 180,
    Category.customer_support: 90,
    Category.travel_planning: 180,
}
DEFAULT_BUDGET = 120


class LatencyScorer:
    name = "latency"

    def score(self, task: Task, trace: Trace) -> ScoreResult:
        budget = BUDGETS.get(task.category, DEFAULT_BUDGET)
        return ScoreResult(self.name, clamp(1 - trace.latency_s / budget), raw=trace.latency_s)
