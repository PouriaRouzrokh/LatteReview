---
name: flagging-items
description: Flags review items for revisiting when information is insufficient or ambiguous. Use when the agent encounters items that cannot be confidently assessed and should be re-reviewed later with updated context.
---

# Flagging Items for Revisit

Flag items that cannot be confidently reviewed due to missing information, ambiguity, or other issues. Flagged items will be re-reviewed at the end of the round with updated context from your memories and prior action logs.

## Available Tools

- `flag_for_revisit(reason)` — Flag the current item for revisiting with a reason
- `list_flagged_items()` — List all flagged items with their reasons and resolution status
- `resolve_current_flag()` — Mark the current item's flag as resolved (use during revisit pass)

## When to Flag

- Abstract or key sections are missing or incomplete
- Conflicting information that needs cross-referencing
- Borderline cases where additional context from other items might help
- Technical terminology or methods you are uncertain about

## When NOT to Flag

- Items you can assess with reasonable confidence
- Items that are simply low quality (score them accordingly instead)
- Items outside the review scope (just note this in your reasoning)

## Tips

- Provide specific, actionable reasons when flagging
- During revisit, check your memories for insights gained from other items
- Resolve flags when you can make a confident assessment on revisit
