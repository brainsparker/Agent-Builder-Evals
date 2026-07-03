from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from agent_evals.gate import Tolerances
from agent_evals.lfd import LfdConfig


class AgentConfig(BaseModel):
    """The agent under test.

    Either point at a built-in provider (optionally overriding its system
    prompt), or supply a ``plugin`` factory that returns your own adapter and
    tool registry. ``plugin`` takes precedence when set.
    """

    provider: str = "anthropic"
    model: str | None = None
    system_prompt: str | None = None
    plugin: str | None = None


class ProjectConfig(BaseModel):
    tasks_dir: Path | None = None
    task_ids: list[str] | None = None
    category: str | None = None
    all_tasks: bool = False
    baseline: Path = Path("results/baseline.json")
    judge_model: str = "claude-haiku-4-5"
    record: str = "live"
    seed: int = 42
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tolerances: Tolerances = Field(default_factory=Tolerances)
    lfd: LfdConfig = Field(default_factory=LfdConfig)


def _resolve(base_dir: Path, value: Path | None) -> Path | None:
    if value is None:
        return None
    return value if value.is_absolute() else (base_dir / value)


def load_project_config(path: Path) -> ProjectConfig:
    """Load ``agentevals.toml``. Paths are resolved relative to the config file."""
    if not path.exists():
        raise FileNotFoundError(
            f"No config at {path}. Create an agentevals.toml (see the example in the repo root)."
        )
    with path.open("rb") as fh:
        payload = tomllib.load(fh)

    # Allow a flat or [project]-nested layout for the top-level keys.
    project = {**payload, **payload.get("project", {})}
    project.pop("project", None)

    cfg = ProjectConfig.model_validate(
        {
            **{k: v for k, v in project.items() if k not in {"agent", "tolerances", "lfd"}},
            "agent": payload.get("agent", {}),
            "tolerances": payload.get("tolerances", {}),
            "lfd": payload.get("lfd", {}),
        }
    )

    base_dir = path.resolve().parent
    cfg.tasks_dir = _resolve(base_dir, cfg.tasks_dir)
    resolved_baseline = _resolve(base_dir, cfg.baseline)
    assert resolved_baseline is not None  # baseline always has a default
    cfg.baseline = resolved_baseline
    for attr in ("state", "goal"):
        resolved = _resolve(base_dir, getattr(cfg.lfd, attr))
        assert resolved is not None  # both always have defaults
        setattr(cfg.lfd, attr, resolved)
    return cfg
