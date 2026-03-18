"""Dependency injection container for AgenticReviewer tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from lattereview.agentic.logging.action_log import ActionLogger
    from lattereview.agentic.memory.store import MemoryStore
    from lattereview.agentic.flags.store import FlagStore
    from lattereview.agentic.helpers.manager import HelperAgentManager
    from lattereview.agentic.skills import SkillRegistry


@dataclass
class ReviewDeps:
    """Runtime dependencies passed to all skill tools via RunContext[ReviewDeps].

    This dataclass is the single dependency injection container that every tool
    receives through Pydantic AI's RunContext mechanism. It carries everything
    a tool needs: the current item being reviewed, file-backed stores for
    memory/flags, the action logger, and references to helper agents.
    """

    item_id: str
    item_text: str
    agent_name: str
    round_id: str
    max_iterations: int
    agentic_effort: str = "medium"
    working_dir: Optional[Path] = None
    memory_store: Optional["MemoryStore"] = None
    flag_store: Optional["FlagStore"] = None
    action_logger: Optional["ActionLogger"] = None
    helper_manager: Optional["HelperAgentManager"] = None
    skill_registry: Optional["SkillRegistry"] = None
    extra: dict[str, Any] = field(default_factory=dict)
