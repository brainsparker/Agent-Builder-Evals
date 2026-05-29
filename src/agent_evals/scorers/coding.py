from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from agent_evals.models import Task, Trace
from agent_evals.scorers.base import ScoreResult


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class CodingHiddenTestsScorer:
    name = "coding_tests"

    def score(self, task: Task, trace: Trace) -> ScoreResult:
        if not task.hidden_tests:
            return ScoreResult(self.name, 1, applicable=False)
        match = CODE_BLOCK_RE.search(trace.final_output)
        code = match.group(1) if match else trace.final_output
        tests_path = Path(__file__).parents[1] / "tasks" / "_fixtures" / "coding" / task.hidden_tests
        with tempfile.TemporaryDirectory(prefix="agent-evals-hidden-") as tmp:
            work = Path(tmp)
            (work / "solution.py").write_text(code, encoding="utf-8")
            (work / "test_solution.py").write_text(tests_path.read_text(encoding="utf-8"), encoding="utf-8")
            proc = subprocess.run(
                ["python", "-m", "pytest", "-q", str(work / "test_solution.py")],
                cwd=work,
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )
        return ScoreResult(
            self.name,
            1.0 if proc.returncode == 0 else 0.0,
            raw={"stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]},
        )
