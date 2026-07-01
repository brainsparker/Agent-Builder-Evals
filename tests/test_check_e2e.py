from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from agent_evals.cli import app

runner = CliRunner()

PLUGIN_SRC = '''
import os
from agent_evals.models import Trace
from agent_evals.plugins import AgentPlugin
from agent_evals.tools import build_default_registry


class StubAdapter:
    provider = "stub-agent"
    model = "stub-1"

    def run_task(self, task, registry, ctx):
        # Quality is read fresh on every call so a "check" that truly re-runs
        # the candidate sees the change; a check that replayed the baseline would not.
        if os.environ.get("STUB_QUALITY") == "low":
            output = ""
        else:
            output = " ".join(task.expected_outcomes)
        return Trace(
            provider="stub-agent",
            model="stub-1",
            final_output=output,
            finished_reason="completed",
        )


def build(cfg):
    return AgentPlugin(adapter=StubAdapter(), registry=build_default_registry())
'''


def _write_project(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "stub_plugin.py").write_text(PLUGIN_SRC, encoding="utf-8")
    config = tmp_path / "agentevals.toml"
    config.write_text(
        """
[project]
baseline = "baseline.json"
record   = "replay"
task_ids = ["company_research_001"]

[agent]
plugin = "stub_plugin:build"

[tolerances]
overall_drop = 0.02
""",
        encoding="utf-8",
    )
    return config, tmp_path / "baseline.json"


def test_update_baseline_then_check_passes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("STUB_QUALITY", raising=False)
    config, baseline = _write_project(tmp_path)
    out = tmp_path / "results"

    pin = runner.invoke(app, ["check", "--update-baseline", "--config", str(config), "--out", str(out)])
    assert pin.exit_code == 0, pin.output
    assert baseline.exists()

    ok = runner.invoke(app, ["check", "--config", str(config), "--out", str(out)])
    assert ok.exit_code == 0, ok.output
    assert "Gate passed" in ok.output


def test_regression_breaches_and_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("STUB_QUALITY", raising=False)
    config, baseline = _write_project(tmp_path)
    out = tmp_path / "results"

    # Pin a good baseline.
    pin = runner.invoke(app, ["check", "--update-baseline", "--config", str(config), "--out", str(out)])
    assert pin.exit_code == 0, pin.output

    # Now degrade the candidate. If `check` re-runs the agent (it must), the gate fails.
    monkeypatch.setenv("STUB_QUALITY", "low")
    bad = runner.invoke(app, ["check", "--config", str(config), "--out", str(out)])
    assert bad.exit_code == 1, bad.output
    assert "BREACH" in bad.output


def test_missing_baseline_errors(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("STUB_QUALITY", raising=False)
    config, _ = _write_project(tmp_path)
    out = tmp_path / "results"
    res = runner.invoke(app, ["check", "--config", str(config), "--out", str(out)])
    assert res.exit_code == 2, res.output
    assert "No baseline" in res.output
