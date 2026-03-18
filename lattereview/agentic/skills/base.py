"""Skill manifest and metadata types for the skills system."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Anthropic Agent Skills spec naming rules
_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9\-]{0,63}$")
_RESERVED_WORDS = {"anthropic", "claude"}

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024


@dataclass
class SkillManifest:
    """Parsed metadata from a SKILL.md frontmatter.

    Follows the Anthropic Agent Skills spec:
    - name: lowercase, hyphens, max 64 chars, no reserved words
    - description: third person, max 1024 chars, includes what + when
    """

    name: str
    description: str
    path: Path
    body: str = ""  # Markdown body below frontmatter (L2/L3 content)
    toolset: Optional[object] = field(default=None, repr=False)  # FunctionToolset, loaded lazily

    def __post_init__(self) -> None:
        validate_skill_name(self.name)
        if len(self.description) > MAX_DESCRIPTION_LENGTH:
            raise ValueError(
                f"Skill '{self.name}' description exceeds {MAX_DESCRIPTION_LENGTH} chars "
                f"({len(self.description)} chars)"
            )

    @property
    def has_toolset(self) -> bool:
        """Whether the skill's toolset has been loaded."""
        return self.toolset is not None

    def to_description_dict(self) -> dict[str, str]:
        """Return L1 metadata for system prompt injection."""
        return {"name": self.name, "description": self.description}


def validate_skill_name(name: str) -> None:
    """Validate a skill name against the Anthropic Agent Skills spec.

    Rules:
    - Max 64 characters
    - Lowercase letters, numbers, and hyphens only
    - No reserved words ("anthropic", "claude")
    """
    if not _SKILL_NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid skill name '{name}': must be 1-64 chars, "
            f"lowercase letters, numbers, and hyphens only, starting with letter or number"
        )
    for word in _RESERVED_WORDS:
        if word in name:
            raise ValueError(f"Invalid skill name '{name}': contains reserved word '{word}'")
