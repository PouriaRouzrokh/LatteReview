"""Flagging tools — flag items for revisiting when assessment is uncertain."""

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from lattereview.agentic.deps import ReviewDeps

toolset = FunctionToolset()


@toolset.tool(retries=3)
async def flag_for_revisit(ctx: RunContext[ReviewDeps], reason: str) -> str:
    """Flag the current item for revisiting at the end of the round.

    FLAG WHEN: The abstract is missing critical information needed for assessment
    (e.g., no methods section, key outcome data absent). DO NOT FLAG: Items that
    are simply low quality — score them low instead. The item will be re-reviewed
    later with updated context from your memories and prior reviews.

    Args:
        ctx: Run context with dependencies.
        reason: Why this item needs revisiting (be specific and actionable).
    """
    store = ctx.deps.flag_store
    if store is None:
        raise ModelRetry(
            "Flag store is not available. Flagging requires a working directory " "to be configured on the reviewer."
        )

    result = await store.add_flag(item_id=ctx.deps.item_id, reason=reason)
    return result


@toolset.tool(retries=3)
async def list_flagged_items(ctx: RunContext[ReviewDeps]) -> str:
    """List all flagged items with their reasons and resolution status.

    Args:
        ctx: Run context with dependencies.
    """
    store = ctx.deps.flag_store
    if store is None:
        raise ModelRetry("Flag store is not available.")

    entries = await store.list_flags()

    if not entries:
        return "No items have been flagged."

    lines = [f"Flagged items ({len(entries)} total):"]
    for entry in entries:
        status = "RESOLVED" if entry["resolved"] else "UNRESOLVED"
        lines.append(f"- [{status}] {entry['item_id']}: {entry['reason']}")

    return "\n".join(lines)


@toolset.tool(retries=3)
async def resolve_current_flag(ctx: RunContext[ReviewDeps]) -> str:
    """Mark the current item's flag as resolved.

    Use during a revisit pass when you can now make a confident assessment.

    Args:
        ctx: Run context with dependencies.
    """
    store = ctx.deps.flag_store
    if store is None:
        raise ModelRetry("Flag store is not available.")

    resolved = await store.resolve_flag(ctx.deps.item_id)

    if resolved:
        return f"Flag for item '{ctx.deps.item_id}' marked as resolved."
    else:
        return f"No unresolved flag found for item '{ctx.deps.item_id}'."
