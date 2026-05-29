from types import SimpleNamespace

from agent_evals.adapters.anthropic_adapter import AnthropicAdapter
from agent_evals.models import Task
from agent_evals.tools import ToolContext, build_default_registry
from agent_evals.tools.search_backend import MockBackend


class FakeMessages:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        if self.calls == 1:
            block = SimpleNamespace(
                type="tool_use",
                id="toolu_1",
                name="web_search",
                input={"query": "x", "max_results": 1},
            )
            return SimpleNamespace(content=[block], usage=usage)
        text = SimpleNamespace(type="text", text="final with x")
        return SimpleNamespace(content=[text], usage=usage)


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()


def test_anthropic_adapter_tool_loop_with_fake_client():
    task = Task(
        id="t",
        category="company_research",
        prompt="find x",
        allowed_tools=["web_search"],
        max_tool_calls=3,
    )
    ctx = ToolContext(search_backend=MockBackend(search_results={"x": "hit"}))
    trace = AnthropicAdapter(client=FakeClient()).run_task(task, build_default_registry(), ctx)
    assert trace.finished_reason == "completed"
    assert trace.final_output == "final with x"
    assert trace.tool_calls[0].name == "web_search"
    assert trace.usage.input_tokens == 20
