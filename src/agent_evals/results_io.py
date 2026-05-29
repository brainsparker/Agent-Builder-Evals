from __future__ import annotations

import json
from pathlib import Path

from agent_evals.models import RunResult


def safe_model_name(model: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in model)


def result_path(out_dir: Path, timestamp: str, provider: str, model: str, git_rev: str | None) -> Path:
    shortgit = (git_rev or "nogit")[:8]
    stamp = timestamp.replace(":", "").replace("-", "").split(".")[0]
    return out_dir / f"run_{stamp}_{provider}_{safe_model_name(model)}_{shortgit}.json"


def write_result(run: RunResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = result_path(
        out_dir,
        run.manifest.utc_timestamp,
        run.manifest.provider,
        run.manifest.model,
        run.manifest.git_revision,
    )
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path


def read_result(path: Path) -> RunResult:
    return RunResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
