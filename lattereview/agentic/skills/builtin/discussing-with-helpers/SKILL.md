---
name: discussing-with-helpers
description: Consults helper agents for expert opinions during review. Use when the agent needs a second opinion, domain expertise, or wants to validate its assessment with another agent.
---

# Discussing with Helper Agents

Consult helper agents when you need expert input, a second opinion, or specialized knowledge that would improve your review quality. Helpers are ordered by priority — start with the first helper and escalate to the next if the response is insufficient.

## Available Tools

- `discuss_with_helper(question, helper_index)` — Ask a helper agent a question about the current item

## When to Consult Helpers

- You need domain-specific expertise outside your specialization
- The item is borderline and a second opinion would increase confidence
- You encounter conflicting information that benefits from another perspective
- Technical details require verification from a specialist

## When NOT to Consult Helpers

- You can assess the item confidently on your own
- The question is trivial or doesn't benefit from another perspective
- You've already consulted a helper about the same question

## Tips

- Ask specific, focused questions — helpers respond better to clear queries
- Include relevant context in your question (what you've found, what's unclear)
- If the first helper's response is insufficient, try the next helper (higher index)
- Helper responses are advisory — you make the final assessment
- Helpers maintain their own memories across consultations
