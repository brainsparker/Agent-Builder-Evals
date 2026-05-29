from __future__ import annotations

import json
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class SearchBackend(Protocol):
    def search(self, query: str, max_results: int = 5) -> str:
        ...

    def extract(self, url: str) -> str:
        ...


class MockBackend:
    def __init__(self, search_results: dict[str, str] | None = None, extracts: dict[str, str] | None = None):
        self.search_results = search_results or {}
        self.extracts = extracts or {}

    def search(self, query: str, max_results: int = 5) -> str:
        return self.search_results.get(
            query,
            f"Mock search results for {query} (max_results={max_results})",
        )

    def extract(self, url: str) -> str:
        return self.extracts.get(url, f"Mock extracted content for {url}")


class YouBackend:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> str:
        params = urlencode({"query": query, "count": max_results})
        request = Request(
            f"https://api.ydc-index.io/v1/search?{params}",
            headers={"X-API-Key": self.api_key, "Accept": "application/json"},
            method="GET",
        )
        return _request_json(request)

    def extract(self, url: str) -> str:
        body = json.dumps({"urls": [url], "formats": ["markdown", "metadata"], "crawl_timeout": 10}).encode()
        request = Request(
            "https://ydc-index.io/v1/contents",
            data=body,
            headers={
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        return _request_json(request)


def _request_json(request: Request) -> str:
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"You.com API HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"You.com API request failed: {exc}") from exc
    parsed = json.loads(payload)
    return json.dumps(parsed, indent=2, sort_keys=True)
