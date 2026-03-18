---
name: searching-semantic-scholar
description: Searches Semantic Scholar for academic papers and citation data. Use when the agent needs to find research papers, check citation counts, or explore paper relationships. Optional S2_API_KEY for higher rate limits.
---

# Semantic Scholar Search

Search Semantic Scholar's academic paper database for research articles and citation data.

## Available Tools

- `search_papers(query, max_results)` — Search for papers by keyword. Returns titles, authors, year, citation count, and paper IDs.
- `get_paper_details(paper_id)` — Get detailed information about a specific paper including abstract, references, and citations.

## Usage Tips

- Accepts Semantic Scholar paper IDs, DOIs, arXiv IDs (prefix with `ARXIV:`), or PMIDs (prefix with `PMID:`)
- Set `S2_API_KEY` environment variable for higher rate limits (optional, works without it)
- Citation counts and influence metrics help assess paper impact
- Requires the `httpx` package

## When to Use

- Finding cited or citing papers for a reference
- Checking citation counts and academic impact
- Searching across disciplines (not limited to biomedical like PubMed)
- Exploring paper relationship graphs
