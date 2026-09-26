#!/usr/bin/env python3
"""
web_search - searches the web through the self-hosted SearXNG instance (CS-21 tool).

Runs in the 'agent-tools' conda env as a tool script (see amadeo_utils/ai/llm/tools/script_tools.py):
the tool server calls it with the arguments on stdin and reads one JSON answer from stdout.

SearXNG is a metasearch engine running on the LAN; it asks several public search engines and merges
their results, so no search API key is needed and no single engine sees this machine directly. Its
address comes from this tool's config - the model only ever chooses the query, never where the
request goes. (SearXNG must have 'json' in search.formats, and its bot limiter off, for this to work.)

Security flags: 'outbound' (the query leaves the network, via SearXNG, to public search engines) and
'untrusted_output' (titles and snippets are written by anyone). In the tool server's default 'auto'
mode, only a delegate WORKER can call it, and what it reads never reaches the main model. The tool
server's credential filter refuses a query shaped like a password or key before this script runs.

Config (web_search.json in your tool config folder - outside the repo):
    {
        "searxng_url": "http://searxng.local:8080",
        "max_results": 10,
        "safesearch": 1              # 0 off, 1 moderate, 2 strict
    }

Usage:
    web_search.py --describe
    echo '{"query": "national weather service api"}' | web_search.py --config web_search.json
"""
import requests

from amadeo_utils.ai.llm.tools.script_tools import ToolAnswer, ToolError, tool_script_main

DEFAULT_RESULTS = 5
HARD_MAX_RESULTS = 10          # the config may lower this, never raise it
MAX_SNIPPET_CHARS = 400        # snippets are for choosing what to fetch, not for reading in full
MAX_QUERY_CHARS = 300
HTTP_TIMEOUT = (5, 15)
TIME_RANGES = ("day", "week", "month", "year")

DEFINITION = {
    "name": "web_search",
    "description": "Searches the web and returns the top results: title, URL and a short snippet for each. "
                   "Use fetch_url on a result to read the page itself.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "max_results": {"type": "integer", "description": f"How many results (1-{HARD_MAX_RESULTS}). Default {DEFAULT_RESULTS}."},
            "time_range": {"type": "string", "description": "Only results from the last 'day', 'week', 'month' or 'year'. Default: any time."},
        },
        "required": ["query"],
    },
    "flags": {"outbound": True, "untrusted_output": True},
    "timeout_s": 25,
}


def handle(arguments, config):
    """The tool: one SearXNG query, trimmed to the fields a model needs to decide what to read."""
    base = (config.get("searxng_url") or "").rstrip("/")
    if not base:
        raise ToolError("web search is not configured (no searxng_url)")

    query = str(arguments.get("query", "")).strip()
    if not query:
        raise ToolError("a query is required")
    if len(query) > MAX_QUERY_CHARS:
        raise ToolError(f"the query is longer than {MAX_QUERY_CHARS} characters")
    try:
        wanted = int(arguments.get("max_results", DEFAULT_RESULTS))
    except (TypeError, ValueError):
        raise ToolError("max_results must be a whole number")
    limit = min(HARD_MAX_RESULTS, int(config.get("max_results", HARD_MAX_RESULTS)))
    wanted = max(1, min(limit, wanted))

    params = {"q": query, "format": "json", "safesearch": int(config.get("safesearch", 1))}
    time_range = arguments.get("time_range")
    if time_range:
        if time_range not in TIME_RANGES:
            raise ToolError(f"time_range must be one of {', '.join(TIME_RANGES)}")
        params["time_range"] = time_range

    try:
        response = requests.get(f"{base}/search", params=params, timeout=HTTP_TIMEOUT,
                                headers={"Accept": "application/json"})
    except requests.RequestException as e:
        raise ToolError(f"the search service could not be reached ({type(e).__name__})")
    if response.status_code == 403:
        raise ToolError("the search service refused the request - is 'json' enabled in SearXNG's search.formats?")
    if response.status_code == 429:
        raise ToolError("the search service is rate-limiting - is SearXNG's bot limiter turned off?")
    if response.status_code != 200:
        raise ToolError(f"the search service answered HTTP {response.status_code}")
    try:
        data = response.json()
    except ValueError:
        raise ToolError("the search service sent something that was not JSON")

    results = []
    for item in data.get("results", []):
        url = item.get("url")
        if not url or not str(url).startswith(("http://", "https://")):
            continue
        snippet = " ".join(str(item.get("content") or "").split())
        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[:MAX_SNIPPET_CHARS - 3] + "..."
        results.append({"title": item.get("title"), "url": url, "snippet": snippet,
                        "published": item.get("publishedDate")})
        if len(results) >= wanted:
            break
    answers = [str(a.get("answer", a)) if isinstance(a, dict) else str(a) for a in data.get("answers", [])][:2]
    if not results and not answers:
        raise ToolError("no results")
    # The history line says what was searched and how much came back - never a title or snippet, which strangers wrote.
    summary = f"{len(results)} result{'s' if len(results) != 1 else ''} for {query!r}"
    return ToolAnswer({"query": query, "results": results, **({"instant_answers": answers} if answers else {})}, summary)


if __name__ == "__main__":
    tool_script_main(DEFINITION, handle)
