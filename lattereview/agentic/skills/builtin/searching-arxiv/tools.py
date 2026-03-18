"""arXiv search tools via the arxiv package.

NOTE: Do NOT use ``from __future__ import annotations`` here.
It turns type hints into strings and breaks Pydantic AI's runtime
resolution of ``RunContext[ReviewDeps]``.
"""

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from lattereview.agentic.deps import ReviewDeps

toolset = FunctionToolset()


def _import_arxiv():
    """Lazy import arxiv with clear error."""
    try:
        import arxiv
    except ImportError:
        raise ModelRetry("arXiv search requires the 'arxiv' package. " "Install with: pip install arxiv")
    return arxiv


@toolset.tool
async def search_arxiv(ctx: RunContext[ReviewDeps], query: str, max_results: int = 5) -> str:
    """Search arXiv for scientific papers matching the query.

    Args:
        ctx: Run context with dependencies.
        query: Search query (supports arXiv query syntax like ti:, au:, cat:).
        max_results: Maximum number of results (default 5, max 20).
    """
    arxiv = _import_arxiv()
    max_results = min(max(1, max_results), 20)

    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        articles = list(client.results(search))
    except Exception as e:
        raise ModelRetry(f"arXiv search failed: {e}")

    if not articles:
        return f"No arXiv results found for: {query}"

    results = []
    for i, article in enumerate(articles):
        title = article.title.replace("\n", " ").strip()
        arxiv_id = article.entry_id.split("/abs/")[-1] if "/abs/" in article.entry_id else article.entry_id
        authors = [a.name for a in article.authors[:3]]
        author_str = ", ".join(authors)
        if len(article.authors) > 3:
            author_str += " et al."
        year = article.published.year if article.published else "Unknown"
        categories = ", ".join(article.categories[:3]) if article.categories else "Unknown"

        results.append(
            f"{i + 1}. **{title}**\n   {author_str} ({year})\n   arXiv: {arxiv_id}\n   Categories: {categories}"
        )

    return f"arXiv results for '{query}':\n\n" + "\n\n".join(results)


@toolset.tool
async def get_paper(ctx: RunContext[ReviewDeps], arxiv_id: str) -> str:
    """Get detailed information about a specific arXiv paper.

    Args:
        ctx: Run context with dependencies.
        arxiv_id: arXiv paper ID (e.g., '2301.12345' or '2301.12345v2').
    """
    arxiv = _import_arxiv()

    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        articles = list(client.results(search))
    except Exception as e:
        raise ModelRetry(f"Failed to retrieve arXiv paper {arxiv_id}: {e}")

    if not articles:
        return f"No paper found for arXiv ID: {arxiv_id}"

    article = articles[0]
    title = article.title.replace("\n", " ").strip()
    authors = [a.name for a in article.authors[:10]]
    author_str = ", ".join(authors)
    if len(article.authors) > 10:
        author_str += f" et al. ({len(article.authors)} total)"
    year = article.published.strftime("%Y-%m-%d") if article.published else "Unknown"
    updated = article.updated.strftime("%Y-%m-%d") if article.updated else None
    categories = ", ".join(article.categories) if article.categories else "Unknown"
    abstract = article.summary.strip() if article.summary else "No abstract available."
    pdf_url = article.pdf_url or ""

    parts = [
        f"**{title}**",
        f"{author_str}",
        f"Published: {year}" + (f" (updated: {updated})" if updated and updated != year else ""),
        f"Categories: {categories}",
    ]
    if pdf_url:
        parts.append(f"PDF: {pdf_url}")
    parts.append(f"\n**Abstract:**\n{abstract}")

    return "\n".join(parts)
