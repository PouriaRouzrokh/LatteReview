"""Skills system — discovery, registry, and lazy loading.

The SkillRegistry is the main entry point for managing skills.

Usage:
    registry = SkillRegistry()
    registry.discover()                           # L1: scan for SKILL.md files
    registry.enable(["searching-content"])         # L2: load toolsets
    toolsets = registry.get_enabled_toolsets()     # For Agent(..., toolsets=...)
    descriptions = registry.get_enabled_descriptions()  # For system prompt
"""

from lattereview.agentic.skills.base import SkillManifest, validate_skill_name
from lattereview.agentic.skills.loader import discover_skills, load_toolset, parse_skill_md
from lattereview.agentic.skills.registry import SkillRegistry

__all__ = [
    "SkillManifest",
    "SkillRegistry",
    "discover_skills",
    "load_toolset",
    "parse_skill_md",
    "validate_skill_name",
]
