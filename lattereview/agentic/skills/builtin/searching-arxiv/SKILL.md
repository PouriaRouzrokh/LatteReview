---
name: searching-arxiv
description: Searches arXiv for preprints and scientific papers. Use when the agent needs to find preprints, check paper details, or retrieve abstracts from arXiv.
---

# arXiv Search

Search arXiv for preprints and scientific papers across physics, mathematics, computer science, and more.

## Available Tools

- `search_arxiv(query, max_results)` — Search arXiv by keyword. Returns titles, authors, categories, and arXiv IDs.
- `get_paper(arxiv_id)` — Get detailed information about a specific arXiv paper including full abstract.

## Usage Tips

- Use arXiv-style queries for best results (e.g., "ti:transformer AND cat:cs.CL")
- arXiv IDs look like `2301.12345` or `2301.12345v2`
- Papers may be preprints and not yet peer-reviewed
- Requires the `arxiv` package

## When to Use

- Finding preprints in CS, physics, math, biology, or other arXiv categories
- Retrieving details for a known arXiv ID
- Searching for the latest research before it appears in journals
