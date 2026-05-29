from __future__ import annotations

from agent_evals.models import Task, Trace
from agent_evals.scorers.base import ScoreResult, clamp


class TaskCompletionScorer:
    name = "completion"

    def score(self, task: Task, trace: Trace) -> ScoreResult:
        output = trace.final_output.lower()
        if trace.finished_reason != "completed" or not output:
            return ScoreResult(self.name, 0, reason="Task did not complete")
        if not task.expected_outcomes:
            return ScoreResult(self.name, 1, reason="No explicit expected outcomes")
        hits = 0
        for outcome in task.expected_outcomes:
            tokens = [token.lower() for token in outcome.replace("-", " ").split() if len(token) > 3]
            if tokens and any(token in output for token in tokens):
                hits += 1
        return ScoreResult(
            self.name,
            clamp(hits / len(task.expected_outcomes)),
            raw={"matched": hits, "total": len(task.expected_outcomes)},
        )
