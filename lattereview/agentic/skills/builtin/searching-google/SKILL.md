---
name: searching-google
description: Searches the web using Google via the Gemini API. Use when the agent needs to find current information, verify claims, or look up background context from the web. Requires GEMINI_API_KEY.
---

# Google Web Search

Search the web using Google's search capabilities via the Gemini API.

## Available Tools

- `google_search(query, max_results)` — Search Google for web results. Returns titles, snippets, and URLs.

## Usage Tips

- Formulate specific, targeted queries rather than broad ones
- Use this for current events, general knowledge, or verifying claims
- Requires the `google-genai` package and `GEMINI_API_KEY` environment variable
- If the API key is missing, consider using `searching-duckduckgo` as a free alternative

## When to Use

- Verifying factual claims in the review item
- Finding background context on unfamiliar topics
- Looking up author affiliations or institutional details
- Checking for retractions or corrections
