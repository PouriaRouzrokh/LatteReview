"""DuckDuckGo web search tools — free, no API key required.

NOTE: Do NOT use ``from __future__ import annotations`` here.
It turns type hints into strings and breaks Pydantic AI's runtime
resolution of ``RunContext[ReviewDeps]``.
"""

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from lattereview.agentic.deps import ReviewDeps

toolset = FunctionToolset()


@toolset.tool
async def duckduckgo_search(ctx: RunContext[ReviewDeps], query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo for web results.

    Args:
        ctx: Run context with dependencies.
        query: The search query string.
        max_results: Maximum number of results to return (default 5, max 10).
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ModelRetry(
            "DuckDuckGo search requires the 'duckduckgo-search' package. " "Install with: pip install duckduckgo-search"
        )

    max_results = min(max(1, max_results), 10)

    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        raise ModelRetry(f"DuckDuckGo search failed: {e}")

    if not raw_results:
        return f"No results found for: {query}"

    results = []
    for i, r in enumerate(raw_results):
        title = r.get("title", "No title")
        body = r.get("body", "No snippet")
        href = r.get("href", "No URL")
        results.append(f"{i + 1}. **{title}**\n   {body}\n   URL: {href}")

    return f"DuckDuckGo results for '{query}':\n\n" + "\n\n".join(results)
