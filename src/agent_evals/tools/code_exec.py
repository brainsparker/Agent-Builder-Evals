from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agent_evals.tools.base import ToolContext, ToolSpec, schema


class CodeExecTool:
    spec = ToolSpec(
        name="code_exec",
        description="Run benchmark-authored Python code in a temporary directory with a timeout.",
        input_schema=schema(
            {
                "language": {"type": "string", "enum": ["python"]},
                "code": {"type": "string", "description": "Python code to execute."},
            },
            ["language", "code"],
        ),
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        if args.get("language") != "python":
            raise ValueError("Only python is supported")
        with tempfile.TemporaryDirectory(prefix="agent-evals-code-") as tmp:
            path = Path(tmp) / "main.py"
            path.write_text(str(args["code"]), encoding="utf-8")
            proc = subprocess.run(
                ["python", str(path)],
                cwd=tmp,
                text=True,
                capture_output=True,
                timeout=8,
                check=False,
            )
        return (
            f"exit_code={proc.returncode}\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
