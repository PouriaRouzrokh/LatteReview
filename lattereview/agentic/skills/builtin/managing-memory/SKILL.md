---
name: managing-memory
description: Saves and retrieves agent memories for cross-item learning. Use when the agent needs to remember patterns, insights, or lessons learned from previous items to inform future reviews.
---

# Memory Management

Save generalizable insights and retrieve them across review items. Memories persist within a round and are available when reviewing subsequent items.

## Available Tools

- `save_memory(title, brief, content)` — Save a new memory with a short title, one-line brief, and full content
- `load_memory(memory_id)` — Load the full content of a specific memory by ID
- `list_memories()` — List all saved memories with their IDs, titles, and briefs
- `load_multiple_memories(memory_ids)` — Load multiple memories at once by their IDs
- `delete_memory(memory_id)` — Delete a memory that is no longer useful

## What to Save

- Generalizable patterns and insights (e.g., "Studies with <50 participants rarely report confidence intervals")
- Review criteria refinements discovered during the review process
- Domain-specific knowledge that applies across items
- Lessons learned from ambiguous or difficult items

## What NOT to Save

- Raw record data, titles, or abstracts from individual items
- Per-item specifics that won't generalize
- Information already in the review instructions

## Tips

- Keep memories concise and actionable
- Use descriptive titles for easy identification
- Delete outdated memories when insights are superseded
- When approaching the memory limit, consolidate related memories
