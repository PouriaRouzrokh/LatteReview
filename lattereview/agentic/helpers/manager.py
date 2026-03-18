"""HelperAgentManager — ordered helper delegation for inter-agent consultation."""

# NOTE: Do NOT use `from __future__ import annotations` here.
# Pydantic AI needs real type objects at runtime for tool function signatures.

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

logger = logging.getLogger(__name__)


class HelperAgentManager:
    """Manages ordered helper agents for inter-agent consultation.

    Helpers are tried in priority order. Each helper gets its own Pydantic AI
    agent run with the reviewing agent's question + item context. Helpers
    never receive the discussing-with-helpers skill (recursion prevention).

    Args:
        helpers: Ordered list of AgenticReviewer instances (first = highest priority).
        helper_max_iterations: Iteration budget for each helper discussion.
    """

    def __init__(
        self,
        helpers: list,
        helper_max_iterations: int = 5,
    ) -> None:
        self._helpers = helpers
        self._helper_max_iterations = helper_max_iterations

    @property
    def helpers(self) -> list:
        """Ordered list of helper reviewers."""
        return self._helpers

    @property
    def helper_count(self) -> int:
        """Number of available helpers."""
        return len(self._helpers)

    @property
    def helper_max_iterations(self) -> int:
        """Iteration budget per helper discussion."""
        return self._helper_max_iterations

    def get_helper_names(self) -> List[str]:
        """Get names of all helpers in priority order."""
        return [h.name for h in self._helpers]

    async def discuss(
        self,
        question: str,
        item_text: str,
        item_id: str = "0",
        round_id: str = "A",
        working_dir: Optional[Path] = None,
        helper_index: int = 0,
    ) -> Tuple[str, str, float]:
        """Consult a helper agent with a question about the current item.

        The helper gets a tailored system prompt with its identity, the item
        context, and any of its own memories. It does NOT get the discussion
        skill (preventing recursion).

        Args:
            question: The reviewing agent's question for the helper.
            item_text: The item text being reviewed.
            item_id: Current item identifier.
            round_id: Current round identifier.
            working_dir: Working directory for helper's memory state.
            helper_index: Which helper to consult (0 = highest priority).

        Returns:
            Tuple of (response_text, helper_name, cost).

        Raises:
            IndexError: If helper_index is out of range.
        """
        if helper_index < 0 or helper_index >= len(self._helpers):
            raise IndexError(
                f"Helper index {helper_index} out of range. "
                f"Available helpers: {len(self._helpers)} "
                f"({', '.join(self.get_helper_names())})"
            )

        helper = self._helpers[helper_index]
        return await self._run_helper(
            helper=helper,
            question=question,
            item_text=item_text,
            item_id=item_id,
            round_id=round_id,
            working_dir=working_dir,
        )

    async def _run_helper(
        self,
        *,
        helper: Any,
        question: str,
        item_text: str,
        item_id: str,
        round_id: str,
        working_dir: Optional[Path],
    ) -> Tuple[str, str, float]:
        """Run a single helper agent consultation.

        The helper gets:
        - Its own identity and backstory
        - A "you are being consulted" framing
        - The reviewing agent's question
        - The item text
        - Its own memory context (if working_dir set)
        - Any skills it has EXCEPT discussing-with-helpers (safety)

        Returns:
            Tuple of (response_text, helper_name, cost).
        """
        from lattereview.agentic.deps import ReviewDeps

        # Build helper's memory context
        memory_store = None
        memory_summaries = None
        if working_dir is not None:
            from lattereview.agentic.memory.store import MemoryStore

            memory_dir = working_dir / f"round_{round_id}" / f"helper_{helper.name}" / "memory"
            memory_store = MemoryStore(memory_dir)
            await memory_store.initialize()

            summaries = await memory_store.get_summaries()
            if summaries:
                memory_summaries = summaries

        # Set up helper skills — exclude discussing-with-helpers to prevent recursion
        toolsets = []
        skill_descriptions = None
        if helper.is_agentic and helper.skills:
            safe_skills = [s for s in helper.skills if s != "discussing-with-helpers"]
            if safe_skills:
                from lattereview.agentic.skills import SkillRegistry

                registry = SkillRegistry()
                registry.discover(*helper.custom_skill_paths)
                registry.enable(safe_skills)
                toolsets = registry.get_enabled_toolsets()
                skill_descriptions = registry.get_enabled_descriptions()
                toolsets.append(registry.build_meta_toolset())

        # Build helper system prompt
        system_prompt = _build_helper_system_prompt(
            helper_name=helper.name,
            helper_backstory=helper.backstory,
            helper_system_prompt=helper.system_prompt,
            memory_summaries=memory_summaries,
            skill_descriptions=skill_descriptions,
        )

        # Build user prompt with question + item context
        user_prompt = (
            f"A reviewing agent has asked for your consultation on the following item.\n\n"
            f"## Question\n{question}\n\n"
            f"## Item Being Reviewed (ID: {item_id})\n{item_text}"
        )

        # Build the Pydantic AI agent — plain text output (no structured output for helpers)
        agent_kwargs: Dict[str, Any] = {
            "deps_type": ReviewDeps,
            "system_prompt": system_prompt,
            "retries": helper.max_retries,
            "output_type": str,
        }

        if helper.model_settings:
            agent_kwargs["model_settings"] = helper.model_settings

        if toolsets and helper.is_agentic:
            agent_kwargs["toolsets"] = toolsets

        agent = Agent(helper.model, **agent_kwargs)

        # Build deps for the helper
        deps = ReviewDeps(
            item_id=item_id,
            item_text=item_text,
            agent_name=helper.name,
            round_id=round_id,
            max_iterations=self._helper_max_iterations,
            agentic_effort=helper.agentic_effort,
            working_dir=working_dir,
            memory_store=memory_store,
        )

        # Set usage limits
        usage_limits = UsageLimits(request_limit=self._helper_max_iterations)

        # Run the helper agent
        result = await agent.run(
            user_prompt,
            deps=deps,
            usage_limits=usage_limits,
        )

        # Extract response text and cost
        response_text = result.output
        cost = 0.0  # Cost tracking deferred to RFD-7

        logger.debug(f"Helper '{helper.name}' responded to question about item {item_id}")
        return response_text, helper.name, cost


def _build_helper_system_prompt(
    *,
    helper_name: str,
    helper_backstory: str,
    helper_system_prompt: str,
    memory_summaries: Optional[List[Dict[str, str]]] = None,
    skill_descriptions: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build the system prompt for a helper agent consultation.

    The prompt frames the helper as a consultant being asked for their
    expertise on a specific question about a review item.
    """
    sections = []

    # Identity
    sections.append(f"# Identity\nYou are {helper_name}.")
    if helper_backstory:
        sections.append(helper_backstory)

    # Role framing
    sections.append(
        "# Your Role\n"
        "You are being consulted by another reviewing agent who needs your expertise. "
        "They will present a question about an item they are reviewing. "
        "Provide a clear, focused, and helpful response to their question. "
        "Draw on your knowledge and any available tools to give the best answer."
    )

    # Custom instructions
    if helper_system_prompt:
        sections.append(f"# Instructions\n{helper_system_prompt}")

    # Skills
    if skill_descriptions:
        lines = ["# Available Skills"]
        for skill in skill_descriptions:
            lines.append(f"- **{skill['name']}**: {skill['description']}")
        sections.append("\n".join(lines))

    # Memory context
    if memory_summaries:
        lines = [f"# Your Memories ({len(memory_summaries)} total)"]
        for mem in memory_summaries:
            lines.append(f"- {mem['id']}: {mem['brief']}")
        lines.append("\nUse the load_memory tool to read full details when relevant.")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)
