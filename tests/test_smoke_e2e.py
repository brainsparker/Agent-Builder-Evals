from types import SimpleNamespace

from agent_evals.models import Task, Trace
from agent_evals.runner import run_benchmark


def test_smoke_runner_with_monkeypatched_adapter(monkeypatch, tmp_path):
    class FakeAdapter:
        def run_task(self, task, registry, ctx):
            return Trace(
                provider="openai",
                model="gpt-5.1",
                final_output="reverse_words code",
                finished_reason="completed",
            )

    monkeypatch.setattr(
        "agent_evals.runner.require_keys",
        lambda *args, **kwargs: SimpleNamespace(openai_api_key="test", anthropic_api_key=None, you_api_key="test"),
    )
    monkeypatch.setattr("agent_evals.runner.YouBackend", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("agent_evals.runner.make_adapter", lambda *_args, **_kwargs: FakeAdapter())
    task = Task(id="smoke", category="coding", prompt="do it", allowed_tools=[])
    path = run_benchmark(
        provider="openai",
        model="gpt-5.1",
        tasks=[task],
        judge_model="claude-haiku-4-5",
        record_mode="live",
        seed=42,
        out_dir=tmp_path,
    )
    assert path.exists()
