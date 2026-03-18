"""Meta-tool for L3 progressive disclosure — get_skill_details.

This module is separate from registry.py to avoid `from __future__ import annotations`
conflicts. Pydantic AI needs real type objects at runtime for tool function type hints.
"""

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from lattereview.agentic.deps import ReviewDeps


def create_meta_toolset(registry: object) -> FunctionToolset:
    """Create a FunctionToolset with the get_skill_details meta-tool.

    Args:
        registry: SkillRegistry instance (typed as object to avoid circular import).

    Returns:
        FunctionToolset with one tool: get_skill_details.
    """
    meta_toolset = FunctionToolset()

    @meta_toolset.tool
    async def get_skill_details(ctx: RunContext[ReviewDeps], skill_name: str) -> str:
        """Get detailed usage instructions for a skill.

        Call this when you need more information about how to use a specific
        skill's tools effectively.

        Args:
            ctx: Run context with dependencies.
            skill_name: Name of the skill to get details for.
        """
        try:
            details = registry.get_skill_details(skill_name)
            if not details:
                return f"No additional details available for skill '{skill_name}'."
            return details
        except ValueError:
            available = ", ".join(registry.enabled_skills)
            return f"Skill '{skill_name}' not found. Available skills: {available}"

    return meta_toolset
