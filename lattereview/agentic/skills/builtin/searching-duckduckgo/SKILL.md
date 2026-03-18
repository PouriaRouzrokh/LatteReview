---
name: searching-duckduckgo
description: Searches the web using DuckDuckGo. Use when the agent needs to find information online without requiring an API key. Free and unlimited.
---

# DuckDuckGo Web Search

Search the web using DuckDuckGo — free, no API key required.

## Available Tools

- `duckduckgo_search(query, max_results)` — Search DuckDuckGo for web results. Returns titles, snippets, and URLs.

## Usage Tips

- Good general-purpose web search for verifying claims and finding context
- No API key needed — works out of the box with the `duckduckgo-search` package
- For academic literature, prefer `searching-pubmed` or `searching-semantic-scholar`
- Keep queries concise and specific for best results

## When to Use

- Verifying factual claims when no API key is available
- Finding background context on topics, authors, or institutions
- General web lookups that don't need academic database precision
