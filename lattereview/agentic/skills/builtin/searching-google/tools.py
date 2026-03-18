"""Google web search tools via Gemini API.

NOTE: Do NOT use ``from __future__ import annotations`` here.
It turns type hints into strings and breaks Pydantic AI's runtime
resolution of ``RunContext[ReviewDeps]``.
"""

import os

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from lattereview.agentic.deps import ReviewDeps

toolset = FunctionToolset()


@toolset.tool
async def google_search(ctx: RunContext[ReviewDeps], query: str, max_results: int = 5) -> str:
    """Search Google for web results related to your query.

    Args:
        ctx: Run context with dependencies.
        query: The search query string.
        max_results: Maximum number of results to return (default 5, max 10).
    """
    try:
        from google import genai
    except ImportError:
        raise ModelRetry(
            "Google search requires the 'google-genai' package. "
            "Install with: pip install google-genai. "
            "Alternatively, use the duckduckgo_search tool which requires no API key."
        )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ModelRetry(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it to use Google search, or use duckduckgo_search instead."
        )

    max_results = min(max(1, max_results), 10)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Search the web for: {query}",
            config=genai.types.GenerateContentConfig(
                tools=[genai.types.Tool(google_search=genai.types.GoogleSearch())],
            ),
        )
    except Exception as e:
        raise ModelRetry(f"Google search failed: {e}")

    # Extract grounding metadata from the response
    results = []
    if response.candidates and response.candidates[0].grounding_metadata:
        metadata = response.candidates[0].grounding_metadata
        chunks = getattr(metadata, "grounding_chunks", None) or []
        for i, chunk in enumerate(chunks[:max_results]):
            web = getattr(chunk, "web", None)
            if web:
                title = getattr(web, "title", "No title") or "No title"
                uri = getattr(web, "uri", "No URL") or "No URL"
                results.append(f"{i + 1}. **{title}**\n   URL: {uri}")

    if not results:
        # Fall back to the text response
        text = response.text if response.text else "No results found."
        return f"Google search for '{query}':\n\n{text}"

    return f"Google search results for '{query}':\n\n" + "\n\n".join(results)
