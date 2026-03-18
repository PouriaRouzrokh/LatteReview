"""Semantic Scholar search tools via httpx.

NOTE: Do NOT use ``from __future__ import annotations`` here.
It turns type hints into strings and breaks Pydantic AI's runtime
resolution of ``RunContext[ReviewDeps]``.
"""

import os

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from lattereview.agentic.deps import ReviewDeps

toolset = FunctionToolset()

_S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
_SEARCH_FIELDS = "paperId,title,authors,year,citationCount,externalIds,abstract"
_DETAILS_FIELDS = "paperId,title,authors,year,citationCount,externalIds,abstract,references,citations,venue,url"


def _get_httpx():
    """Lazy import httpx."""
    try:
        import httpx
    except ImportError:
        raise ModelRetry("Semantic Scholar search requires the 'httpx' package. " "Install with: pip install httpx")
    return httpx


def _s2_headers():
    """Build request headers with optional API key."""
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


@toolset.tool
async def search_papers(ctx: RunContext[ReviewDeps], query: str, max_results: int = 5) -> str:
    """Search Semantic Scholar for academic papers.

    Args:
        ctx: Run context with dependencies.
        query: Search query for finding papers.
        max_results: Maximum number of results (default 5, max 20).
    """
    httpx = _get_httpx()
    max_results = min(max(1, max_results), 20)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{_S2_API_BASE}/paper/search",
                params={"query": query, "limit": max_results, "fields": _SEARCH_FIELDS},
                headers=_s2_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        raise ModelRetry(f"Semantic Scholar search failed: {e}")

    papers = data.get("data", [])
    if not papers:
        return f"No Semantic Scholar results for: {query}"

    results = []
    for i, paper in enumerate(papers):
        title = paper.get("title", "No title")
        paper_id = paper.get("paperId", "Unknown")
        year = paper.get("year", "Unknown")
        citations = paper.get("citationCount", 0)
        authors_raw = paper.get("authors", []) or []
        authors = [a.get("name", "") for a in authors_raw[:3]]
        author_str = ", ".join(authors)
        if len(authors_raw) > 3:
            author_str += " et al."

        ext_ids = paper.get("externalIds", {}) or {}
        doi = ext_ids.get("DOI", "")
        doi_str = f"\n   DOI: {doi}" if doi else ""

        results.append(
            f"{i + 1}. **{title}**\n   {author_str} ({year}) — {citations} citations" f"\n   S2 ID: {paper_id}{doi_str}"
        )

    total = data.get("total", len(papers))
    header = f"Semantic Scholar results for '{query}' ({total} total):\n\n"
    return header + "\n\n".join(results)


@toolset.tool
async def get_paper_details(ctx: RunContext[ReviewDeps], paper_id: str) -> str:
    """Get detailed information about a specific paper from Semantic Scholar.

    Accepts Semantic Scholar IDs, DOIs, arXiv IDs (prefix with ARXIV:),
    or PMIDs (prefix with PMID:).

    Args:
        ctx: Run context with dependencies.
        paper_id: Paper identifier (S2 ID, DOI, ARXIV:id, or PMID:id).
    """
    httpx = _get_httpx()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{_S2_API_BASE}/paper/{paper_id}",
                params={"fields": _DETAILS_FIELDS},
                headers=_s2_headers(),
            )
            resp.raise_for_status()
            paper = resp.json()
    except Exception as e:
        raise ModelRetry(f"Failed to get paper details for '{paper_id}': {e}")

    title = paper.get("title", "No title")
    year = paper.get("year", "Unknown")
    venue = paper.get("venue", "Unknown venue") or "Unknown venue"
    citations = paper.get("citationCount", 0)
    url = paper.get("url", "")

    authors_raw = paper.get("authors", []) or []
    authors = [a.get("name", "") for a in authors_raw[:5]]
    author_str = ", ".join(authors)
    if len(authors_raw) > 5:
        author_str += f" et al. ({len(authors_raw)} total)"

    abstract = paper.get("abstract", "No abstract available.") or "No abstract available."

    refs = paper.get("references", []) or []
    ref_titles = [r.get("title", "Unknown") for r in refs[:5] if r.get("title")]
    refs_str = ""
    if ref_titles:
        refs_str = "\n\n**Key References:**\n" + "\n".join(f"  - {t}" for t in ref_titles)
        if len(refs) > 5:
            refs_str += f"\n  ... and {len(refs) - 5} more"

    cites = paper.get("citations", []) or []
    cite_titles = [c.get("title", "Unknown") for c in cites[:5] if c.get("title")]
    cites_str = ""
    if cite_titles:
        cites_str = "\n\n**Cited By:**\n" + "\n".join(f"  - {t}" for t in cite_titles)
        if len(cites) > 5:
            cites_str += f"\n  ... and {len(cites) - 5} more"

    parts = [
        f"**{title}**",
        f"{author_str} ({year}) — {venue}",
        f"Citations: {citations}",
    ]
    if url:
        parts.append(f"URL: {url}")
    parts.append(f"\n**Abstract:**\n{abstract}")

    return "\n".join(parts) + refs_str + cites_str
