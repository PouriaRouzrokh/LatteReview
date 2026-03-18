"""PubMed search tools via pymed.

NOTE: Do NOT use ``from __future__ import annotations`` here.
It turns type hints into strings and breaks Pydantic AI's runtime
resolution of ``RunContext[ReviewDeps]``.
"""

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from lattereview.agentic.deps import ReviewDeps

toolset = FunctionToolset()

_TOOL_EMAIL = "lattereview@example.com"


def _import_pymed():
    """Lazy import pymed with clear error."""
    try:
        from pymed import PubMed
    except ImportError:
        raise ModelRetry(
            "PubMed search requires the 'pymed' package. "
            "Install with: pip install pymed. "
            "Alternatively, use duckduckgo_search for general web search."
        )
    return PubMed


@toolset.tool
async def search_pubmed(ctx: RunContext[ReviewDeps], query: str, max_results: int = 5) -> str:
    """Search PubMed for biomedical articles matching the query.

    Args:
        ctx: Run context with dependencies.
        query: Search query for PubMed (supports MeSH terms).
        max_results: Maximum number of results (default 5, max 20).
    """
    PubMed = _import_pymed()
    max_results = min(max(1, max_results), 20)

    try:
        pubmed = PubMed(tool="LatteReview", email=_TOOL_EMAIL)
        articles = list(pubmed.query(query, max_results=max_results))
    except Exception as e:
        raise ModelRetry(f"PubMed search failed: {e}")

    if not articles:
        return f"No PubMed results found for: {query}"

    results = []
    for i, article in enumerate(articles):
        pmid = getattr(article, "pubmed_id", "Unknown")
        # pubmed_id can contain multiple IDs separated by newlines
        if pmid and "\n" in str(pmid):
            pmid = str(pmid).split("\n")[0].strip()
        title = getattr(article, "title", "No title") or "No title"
        journal = getattr(article, "journal", "Unknown journal") or "Unknown journal"
        year = getattr(article, "publication_date", None)
        year_str = str(year.year) if year and hasattr(year, "year") else str(year) if year else "Unknown"
        authors_raw = getattr(article, "authors", []) or []
        authors = []
        for a in authors_raw[:3]:
            if isinstance(a, dict):
                name = f"{a.get('firstname', '')} {a.get('lastname', '')}".strip()
                if name:
                    authors.append(name)
            else:
                authors.append(str(a))
        author_str = ", ".join(authors)
        if len(authors_raw) > 3:
            author_str += " et al."

        results.append(f"{i + 1}. **{title}**\n   PMID: {pmid}\n   {author_str} ({year_str}) — {journal}")

    return f"PubMed results for '{query}':\n\n" + "\n\n".join(results)


@toolset.tool
async def get_abstract(ctx: RunContext[ReviewDeps], pmid: str) -> str:
    """Retrieve the full abstract for a PubMed article by PMID.

    Args:
        ctx: Run context with dependencies.
        pmid: PubMed ID of the article.
    """
    PubMed = _import_pymed()

    try:
        pubmed = PubMed(tool="LatteReview", email=_TOOL_EMAIL)
        articles = list(pubmed.query(pmid, max_results=1))
    except Exception as e:
        raise ModelRetry(f"Failed to retrieve abstract for PMID {pmid}: {e}")

    if not articles:
        return f"No article found for PMID: {pmid}"

    article = articles[0]
    title = getattr(article, "title", "No title") or "No title"
    abstract = getattr(article, "abstract", None)

    if not abstract:
        return f"**{title}** (PMID: {pmid})\n\nNo abstract available."

    return f"**{title}** (PMID: {pmid})\n\n{abstract}"


@toolset.tool
async def get_full_text(ctx: RunContext[ReviewDeps], pmid: str) -> str:
    """Attempt to retrieve full text content for a PubMed article.

    Only available for open-access articles. Falls back to abstract if full text
    is not accessible.

    Args:
        ctx: Run context with dependencies.
        pmid: PubMed ID of the article.
    """
    PubMed = _import_pymed()

    try:
        pubmed = PubMed(tool="LatteReview", email=_TOOL_EMAIL)
        articles = list(pubmed.query(pmid, max_results=1))
    except Exception as e:
        raise ModelRetry(f"Failed to retrieve article for PMID {pmid}: {e}")

    if not articles:
        return f"No article found for PMID: {pmid}"

    article = articles[0]
    title = getattr(article, "title", "No title") or "No title"

    # pymed may expose full text via the xml attribute
    xml_content = getattr(article, "xml", None)
    if xml_content:
        # Try to extract body text from XML
        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(str(xml_content)) if isinstance(xml_content, str) else xml_content
            body_parts = []
            for elem in root.iter():
                if elem.text and elem.tag in ("AbstractText", "p", "sec", "body"):
                    body_parts.append(elem.text.strip())
            if body_parts:
                text = "\n\n".join(body_parts)
                return f"**{title}** (PMID: {pmid})\n\nFull text extract:\n\n{text[:5000]}"
        except Exception:
            pass

    # Fall back to abstract
    abstract = getattr(article, "abstract", None)
    if abstract:
        return f"**{title}** (PMID: {pmid})\n\nFull text not available. Abstract:\n\n{abstract}"

    return f"**{title}** (PMID: {pmid})\n\nNeither full text nor abstract available."
