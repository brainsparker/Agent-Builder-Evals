from __future__ import annotations

import re
from urllib.parse import urlparse

from agent_evals.models import Task, Trace
from agent_evals.scorers.base import ScoreResult, clamp


URL_RE = re.compile(r"https?://[^\s)\]]+")


class CitationQualityScorer:
    name = "citation"

    def score(self, task: Task, trace: Trace) -> ScoreResult:
        if not task.reference_citations:
            return ScoreResult(self.name, 1, applicable=False)
        output_urls = set(URL_RE.findall(trace.final_output))
        fetched = " ".join(call.result for call in trace.tool_calls if call.name in {"web_search", "fetch_url"})
        trace_urls = set(URL_RE.findall(fetched)) | {
            str(call.arguments.get("url")) for call in trace.tool_calls if call.name == "fetch_url"
        }
        domain_hits = 0
        for citation in task.reference_citations:
            domain = citation.must_cite_domain
            if domain and any(domain in urlparse(url).netloc for url in output_urls):
                domain_hits += 1
        grounded = len(output_urls & trace_urls)
        score = 0.0
        if output_urls:
            score += 0.5 * grounded / len(output_urls)
        if task.reference_citations:
            score += 0.5 * domain_hits / len(task.reference_citations)
        return ScoreResult(
            self.name,
            clamp(score),
            raw={"output_urls": sorted(output_urls), "grounded": grounded, "domain_hits": domain_hits},
        )
