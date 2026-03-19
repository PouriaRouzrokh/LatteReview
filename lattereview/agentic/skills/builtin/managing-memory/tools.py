"""Memory management tools — save, load, list, and delete agent memories."""

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from lattereview.agentic.deps import ReviewDeps

toolset = FunctionToolset()


@toolset.tool(retries=3)
async def save_memory(
    ctx: RunContext[ReviewDeps],
    title: str,
    brief: str,
    content: str,
) -> str:
    """Save a new memory for future reference across review items.

    SAVE: Generalizable patterns and insights (e.g., 'Radiomics != deep learning',
    'Studies without control groups scored low'). DO NOT SAVE: Per-item facts or
    information already stated in the review instructions.

    Args:
        ctx: Run context with dependencies.
        title: Short descriptive title (e.g., "Small sample size pattern").
        brief: One-line summary for quick scanning.
        content: Full memory content in markdown.
    """
    store = ctx.deps.memory_store
    if store is None:
        raise ModelRetry(
            "Memory store is not available. Memory requires a working directory " "to be configured on the reviewer."
        )

    result = await store.save(title=title, brief=brief, content=content)

    if result.startswith("ERROR:"):
        raise ModelRetry(result)

    return f"Memory saved as {result}: {title}"


@toolset.tool(retries=3)
async def load_memory(ctx: RunContext[ReviewDeps], memory_id: str) -> str:
    """Load the full content of a specific memory.

    Use when a memory listed in your context is relevant to the current item
    and you need the full details to inform your assessment.

    Args:
        ctx: Run context with dependencies.
        memory_id: The memory ID to load (e.g., "mem_001").
    """
    store = ctx.deps.memory_store
    if store is None:
        raise ModelRetry("Memory store is not available.")

    content = await store.load(memory_id)
    if content is None:
        return f"Memory '{memory_id}' not found. Use list_memories to see available memories."

    return content


@toolset.tool(retries=3)
async def list_memories(ctx: RunContext[ReviewDeps]) -> str:
    """List all saved memories with their IDs, titles, and briefs.

    Args:
        ctx: Run context with dependencies.
    """
    store = ctx.deps.memory_store
    if store is None:
        raise ModelRetry("Memory store is not available.")

    entries = await store.list_memories()

    if not entries:
        return "No memories saved yet."

    lines = [f"Memories ({len(entries)} total):"]
    for entry in entries:
        lines.append(f"- {entry['id']}: [{entry['title']}] {entry['brief']}")

    return "\n".join(lines)


@toolset.tool(retries=3)
async def load_multiple_memories(ctx: RunContext[ReviewDeps], memory_ids: str) -> str:
    """Load multiple memories at once.

    Args:
        ctx: Run context with dependencies.
        memory_ids: Comma-separated memory IDs (e.g., "mem_001,mem_003").
    """
    store = ctx.deps.memory_store
    if store is None:
        raise ModelRetry("Memory store is not available.")

    ids = [mid.strip() for mid in memory_ids.split(",") if mid.strip()]
    if not ids:
        return "No memory IDs provided."

    results = await store.load_multiple(ids)

    lines = []
    for mid, content in results.items():
        if content is None:
            lines.append(f"--- {mid}: NOT FOUND ---")
        else:
            lines.append(f"--- {mid} ---\n{content}")

    return "\n\n".join(lines)


@toolset.tool(retries=3)
async def delete_memory(ctx: RunContext[ReviewDeps], memory_id: str) -> str:
    """Delete a memory that is no longer useful.

    Args:
        ctx: Run context with dependencies.
        memory_id: The memory ID to delete (e.g., "mem_001").
    """
    store = ctx.deps.memory_store
    if store is None:
        raise ModelRetry("Memory store is not available.")

    deleted = await store.delete(memory_id)

    if deleted:
        return f"Memory '{memory_id}' deleted."
    else:
        return f"Memory '{memory_id}' not found. Use list_memories to see available memories."
