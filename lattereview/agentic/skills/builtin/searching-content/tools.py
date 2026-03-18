"""Content search tools — regex and keyword search within review item text.

NOTE: Do NOT use ``from __future__ import annotations`` here.
It turns type hints into strings and breaks Pydantic AI's runtime
resolution of ``RunContext[ReviewDeps]``.
"""

import re

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from lattereview.agentic.deps import ReviewDeps

toolset = FunctionToolset()

# Max context chars around each match
_CONTEXT_CHARS = 100


@toolset.tool
async def regex_search(ctx: RunContext[ReviewDeps], pattern: str) -> str:
    """Search the item text with a Python regex pattern.

    Returns all matches with surrounding context. Case-insensitive.

    Args:
        ctx: Run context with dependencies.
        pattern: Python regular expression pattern to search for.
    """
    text = ctx.deps.item_text
    try:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    if not matches:
        return f"No matches found for pattern: {pattern}"

    results = []
    for i, match in enumerate(matches[:20]):  # Cap at 20 matches
        start = max(0, match.start() - _CONTEXT_CHARS)
        end = min(len(text), match.end() + _CONTEXT_CHARS)
        context = text[start:end]
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        results.append(f"Match {i + 1}: '{match.group()}'\n  Context: {context}")

    return f"Found {len(matches)} match(es):\n\n" + "\n\n".join(results)


@toolset.tool
async def keyword_search(ctx: RunContext[ReviewDeps], keywords: str) -> str:
    """Search for keywords in the item text.

    Returns sentences containing any of the specified keywords. Case-insensitive.

    Args:
        ctx: Run context with dependencies.
        keywords: Comma-separated list of keywords to search for.
    """
    text = ctx.deps.item_text
    keyword_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]

    if not keyword_list:
        return "No keywords provided."

    # Split into sentences (rough heuristic)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    matching_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        matched_keywords = [kw for kw in keyword_list if kw in sentence_lower]
        if matched_keywords:
            matching_sentences.append(f"[{', '.join(matched_keywords)}] {sentence.strip()}")

    if not matching_sentences:
        return f"No sentences found containing: {', '.join(keyword_list)}"

    # Cap output
    shown = matching_sentences[:20]
    result = f"Found {len(matching_sentences)} sentence(s) matching keywords:\n\n"
    result += "\n\n".join(shown)
    if len(matching_sentences) > 20:
        result += f"\n\n... and {len(matching_sentences) - 20} more."

    return result
