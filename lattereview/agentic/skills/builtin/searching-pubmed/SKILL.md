---
name: searching-pubmed
description: Searches PubMed for biomedical literature and retrieves abstracts. Use when the agent needs to find medical articles, verify citations, or retrieve PMIDs and abstracts for review items.
---

# PubMed Search

Search PubMed for biomedical and life sciences literature.

## Available Tools

- `search_pubmed(query, max_results)` — Search PubMed by keyword. Returns article titles, authors, journal, year, and PMIDs.
- `get_abstract(pmid)` — Retrieve the full abstract for a specific PMID.
- `get_full_text(pmid)` — Attempt to retrieve full text content for a PMID (availability depends on open access status).

## Usage Tips

- Use MeSH terms when possible for more precise results (e.g., "randomized controlled trial[pt]")
- Start with `search_pubmed` to find relevant articles, then use `get_abstract` for details
- Full text is only available for open-access articles
- Requires the `pymed` package

## When to Use

- Verifying cited references in biomedical papers
- Finding related studies on a medical topic
- Checking publication history of authors
- Looking up methodology details from referenced studies
