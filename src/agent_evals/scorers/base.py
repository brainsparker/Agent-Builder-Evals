from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent_evals.models import Task, Trace


@dataclass(frozen=True)
class ScoreResult:
    name: str
    score: float
    raw: object = None
    reason: str = ""
    applicable: bool = True


class Scorer(Protocol):
    name: str

    def score(self, task: Task, trace: Trace) -> ScoreResult:
        ...


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
