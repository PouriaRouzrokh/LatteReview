---
name: searching-content
description: Searches within the review item text using regex and keyword matching. Use when the agent needs to find specific terms, patterns, or sections within the document being reviewed.
---

# Content Search

Search within the current review item's text for specific patterns or keywords.

## Available Tools

- `regex_search(pattern)` — Search item text with a Python regex pattern. Returns all matches with surrounding context.
- `keyword_search(keywords)` — Search for one or more keywords (comma-separated). Returns sentences containing any keyword.

## Usage Tips

- Use `regex_search` for structured patterns (e.g., "p\s*[<>=]\s*0\.\d+" for p-values)
- Use `keyword_search` for simple term matching (e.g., "randomized, placebo, double-blind")
- Both tools search the full item text available in the current review context
- Searches are case-insensitive by default
