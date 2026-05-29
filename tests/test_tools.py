from agent_evals.tools import ToolContext, build_default_registry
from agent_evals.tools.search_backend import MockBackend


def test_tools_run_and_record_cassette():
    registry = build_default_registry()
    ctx = ToolContext(search_backend=MockBackend(search_results={"q": "https://example.com hit"}))
    result = registry.run("web_search", {"query": "q", "max_results": 1}, ctx)
    assert "example.com" in result
    assert len(ctx.tool_calls) == 1
    assert ctx.cassette


def test_support_lookup_refund():
    registry = build_default_registry()
    ctx = ToolContext(search_backend=MockBackend())
    result = registry.run(
        "support_lookup",
        {"action": "issue_refund", "args": {"order_id": "ord_2001", "amount": 10}},
        ctx,
    )
    assert '"issued": true' in result


def test_provider_schema_source_is_identical():
    registry = build_default_registry()
    specs = [spec.model_dump() for spec in registry.specs(["web_search", "fetch_url"])]
    anthropic = [
        {"name": spec["name"], "description": spec["description"], "input_schema": spec["input_schema"]}
        for spec in specs
    ]
    openai = [
        {"name": spec["name"], "description": spec["description"], "input_schema": spec["input_schema"]}
        for spec in specs
    ]
    assert anthropic == openai
