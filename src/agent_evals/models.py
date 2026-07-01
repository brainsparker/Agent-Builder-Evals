from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Category(str, Enum):
    company_research = "company_research"
    competitive_analysis = "competitive_analysis"
    data_extraction = "data_extraction"
    travel_planning = "travel_planning"
    customer_support = "customer_support"
    coding = "coding"
    knowledge_retrieval = "knowledge_retrieval"


FinishedReason = Literal["completed", "error", "timeout", "max_tool_calls"]
# Built-in providers are "anthropic"/"openai", but bring-your-own agents may
# report any label, so this is an open string rather than a closed literal.
ProviderName = str


class ExpectedToolCall(BaseModel):
    tool: str
    required: bool = True
    min_calls: int = 1
    max_calls: int | None = None
    arg_contains: dict[str, Any] = Field(default_factory=dict)


class ReferenceCitation(BaseModel):
    claim: str
    supporting_url: str | None = None
    must_cite_domain: str | None = None


class ScoringConfig(BaseModel):
    weights: dict[str, float] = Field(default_factory=dict)
    rubric: str = ""
    pass_threshold: float = 0.7


class Task(BaseModel):
    id: str
    category: Category
    prompt: str
    description: str = ""
    expected_outcomes: list[str] = Field(default_factory=list)
    expected_tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    reference_answer: str = ""
    reference_citations: list[ReferenceCitation] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    timeout_s: float = 120
    max_tool_calls: int = 8
    hidden_tests: str | None = None
    output_format: Literal["text", "json", "code"] = "text"

    @field_validator("id")
    @classmethod
    def id_is_slug(cls, value: str) -> str:
        if not value or any(ch.isspace() for ch in value):
            raise ValueError("task id must be a non-empty slug")
        return value


class NormMessage(BaseModel):
    role: str
    content: Any


class NormToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    is_error: bool = False
    started_at: str
    duration_s: float


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    llm_calls: int = 0

    def add(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            llm_calls=self.llm_calls + other.llm_calls,
        )


class Trace(BaseModel):
    provider: ProviderName
    model: str
    messages: list[NormMessage] = Field(default_factory=list)
    tool_calls: list[NormToolCall] = Field(default_factory=list)
    final_output: str = ""
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0
    error: str | None = None
    finished_reason: FinishedReason = "completed"


class DimensionScore(BaseModel):
    score: float = Field(ge=0, le=1)
    raw: Any = None
    weight: float = 0
    reason: str = ""


class Scorecard(BaseModel):
    task_id: str
    category: Category
    provider: ProviderName
    model: str
    dimensions: dict[str, DimensionScore]
    overall: float = Field(ge=0, le=1)
    passed: bool
    cost_usd: float | None = None
    latency_s: float
    failed: bool
    trace: Trace


class RunManifest(BaseModel):
    schema_version: str
    package_version: str
    git_revision: str | None = None
    utc_timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    provider: ProviderName
    model: str
    judge_model: str
    seed: int
    tool_layer_version: str
    pricing_hash: str
    task_hashes: dict[str, str]
    record_mode: Literal["live", "replay"] = "live"
    warnings: list[str] = Field(default_factory=list)


class RunSummary(BaseModel):
    task_count: int
    overall_mean: float
    passed: int
    failure_rate: float
    total_cost_usd: float | None
    median_latency_s: float
    by_dimension: dict[str, float]
    by_category: dict[str, float]


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    manifest: RunManifest
    summary: RunSummary
    scorecards: list[Scorecard]
    cassette: dict[str, Any] = Field(default_factory=dict)
