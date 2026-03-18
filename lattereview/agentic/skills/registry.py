"""SkillRegistry — central manager for skill discovery, enabling, and toolset access."""

# NOTE: Do NOT use `from __future__ import annotations` here.
# Pydantic AI needs real type objects at runtime for tool function signatures.

import logging
from pathlib import Path
from typing import Optional

from pydantic_ai.toolsets import FunctionToolset

from lattereview.agentic.skills.base import SkillManifest
from lattereview.agentic.skills.loader import discover_skills, load_toolset

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Central registry for discovering, enabling, and managing skills.

    Follows the 3-level progressive disclosure model:
    - L1 (discover): Parse SKILL.md frontmatter for all available skills (~100 tokens each)
    - L2 (enable): Import tools.py and load FunctionToolset for selected skills
    - L3 (on-demand): Full SKILL.md body available via get_skill_details()
    """

    def __init__(self) -> None:
        self._available: dict[str, SkillManifest] = {}
        self._enabled: set[str] = set()
        self._discovered: bool = False

    def discover(self, *custom_paths: Path) -> list[str]:
        """Scan builtin and custom directories for skills (L1).

        Args:
            *custom_paths: Additional directories to scan for skill folders.

        Returns:
            List of discovered skill names.
        """
        manifests = discover_skills(*custom_paths)
        self._available = {m.name: m for m in manifests}
        self._discovered = True
        names = list(self._available.keys())
        logger.debug(f"Discovered {len(names)} skills: {names}")
        return names

    @property
    def available_skills(self) -> list[str]:
        """Names of all discovered skills."""
        return list(self._available.keys())

    @property
    def enabled_skills(self) -> list[str]:
        """Names of all enabled skills."""
        return list(self._enabled)

    def get_manifest(self, skill_name: str) -> Optional[SkillManifest]:
        """Get a skill's manifest by name."""
        return self._available.get(skill_name)

    def enable(self, skill_names: list[str]) -> list[str]:
        """Enable skills by name, loading their toolsets (L2).

        Args:
            skill_names: List of skill names to enable.

        Returns:
            List of successfully enabled skill names.

        Raises:
            ValueError: If a skill name is not found in discovered skills.
        """
        if not self._discovered:
            self.discover()

        enabled = []
        for name in skill_names:
            if name not in self._available:
                raise ValueError(f"Skill '{name}' not found. Available: {self.available_skills}")

            manifest = self._available[name]

            # Load toolset if not already loaded
            if not manifest.has_toolset:
                try:
                    manifest.toolset = load_toolset(manifest)
                except (FileNotFoundError, AttributeError, ImportError) as e:
                    logger.error(f"Failed to load skill '{name}': {e}")
                    raise

            self._enabled.add(name)
            enabled.append(name)

        return enabled

    def disable(self, skill_name: str) -> None:
        """Disable a skill (does not unload the toolset)."""
        self._enabled.discard(skill_name)

    def disable_all(self) -> None:
        """Disable all skills."""
        self._enabled.clear()

    def get_enabled_toolsets(self) -> list[FunctionToolset]:
        """Get FunctionToolset objects for all enabled skills.

        Returns:
            List of FunctionToolset instances ready for Agent(..., toolsets=...).
        """
        toolsets = []
        for name in self._enabled:
            manifest = self._available[name]
            if manifest.toolset is not None:
                toolsets.append(manifest.toolset)
        return toolsets

    def get_enabled_descriptions(self) -> list[dict[str, str]]:
        """Get L1 descriptions for enabled skills (for system prompt injection).

        Returns:
            List of {"name": ..., "description": ...} dicts.
        """
        return [self._available[name].to_description_dict() for name in self._enabled if name in self._available]

    def get_skill_details(self, skill_name: str) -> str:
        """Get full SKILL.md body content for a skill (L3).

        This is used by the meta-tool `get_skill_details` that agents
        can call when they need detailed usage instructions.

        Args:
            skill_name: Name of the skill.

        Returns:
            The markdown body of the SKILL.md (below frontmatter).

        Raises:
            ValueError: If skill not found.
        """
        if skill_name not in self._available:
            raise ValueError(f"Skill '{skill_name}' not found. Available: {self.available_skills}")
        return self._available[skill_name].body

    def build_meta_toolset(self) -> FunctionToolset:
        """Create the meta-tool FunctionToolset for L3 progressive disclosure.

        Returns a toolset with a `get_skill_details` tool that agents can call
        to retrieve full SKILL.md instructions for any enabled skill.
        """
        from lattereview.agentic.skills._meta_tool import create_meta_toolset

        return create_meta_toolset(self)
