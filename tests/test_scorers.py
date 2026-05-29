from agent_evals.models import NormToolCall, Task, Trace, Usage
from agent_evals.scorers.aggregate import score_trace


def test_score_trace_deterministic():
    task = Task(
        id="t1",
        category="company_research",
        prompt="Research Example",
        expected_outcomes=["Example"],
        expected_tool_calls=[{"tool": "web_search", "min_calls": 1}],
        allowed_tools=["web_search"],
    )
    trace = Trace(
        provider="openai",
        model="gpt-5.1",
        final_output="Example answer",
        usage=Usage(input_tokens=100, output_tokens=50, llm_calls=1),
        tool_calls=[
            NormToolCall(
                name="web_search",
                arguments={"query": "Example"},
                result="https://example.com",
                started_at="2026-05-29T00:00:00Z",
                duration_s=0.1,
            )
        ],
    )
    card = score_trace(task, trace)
    assert card.overall > 0.5
    assert card.dimensions["tool"].score == 1
