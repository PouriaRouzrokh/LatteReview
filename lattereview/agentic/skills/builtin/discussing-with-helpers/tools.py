"""Discussion tools — consult helper agents for expert opinions during review."""

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from lattereview.agentic.deps import ReviewDeps

toolset = FunctionToolset()


@toolset.tool
async def discuss_with_helper(
    ctx: RunContext[ReviewDeps],
    question: str,
    helper_index: int = 0,
) -> str:
    """Ask a helper agent for their expert opinion on a question about the current item.

    Helpers are ordered by priority (index 0 = highest priority). If the first
    helper's response is insufficient, try the next helper with a higher index.

    Args:
        ctx: Run context with dependencies.
        question: Your question for the helper agent. Be specific and include
            relevant context about what you've found and what's unclear.
        helper_index: Which helper to consult (0 = first/highest priority,
            1 = second, etc.). Default is 0.
    """
    manager = ctx.deps.helper_manager
    if manager is None:
        raise ModelRetry(
            "No helper agents are configured. Helper discussion requires "
            "the reviewer to have helpers=[...] configured."
        )

    if manager.helper_count == 0:
        raise ModelRetry("No helper agents available.")

    if helper_index < 0 or helper_index >= manager.helper_count:
        available = ", ".join(f"{i}: {name}" for i, name in enumerate(manager.get_helper_names()))
        raise ModelRetry(f"Helper index {helper_index} out of range. " f"Available helpers: {available}")

    try:
        response_text, helper_name, cost = await manager.discuss(
            question=question,
            item_text=ctx.deps.item_text,
            item_id=ctx.deps.item_id,
            round_id=ctx.deps.round_id,
            working_dir=ctx.deps.working_dir,
            helper_index=helper_index,
        )
    except Exception as e:
        raise ModelRetry(
            f"Error consulting helper: {e}. " f"Try rephrasing your question or consulting a different helper."
        )

    return f"[Response from helper '{helper_name}']\n\n{response_text}"
