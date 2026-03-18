"""AgenticReviewer — the core agentic review agent wrapping Pydantic AI."""

from __future__ import annotations

import asyncio
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from lattereview.agentic.deps import ReviewDeps
from lattereview.agentic.prompts import build_system_prompt, build_task_prompt
from lattereview.agentic.output_models import ScoringOutput

logger = logging.getLogger(__name__)

# Default task prompt template
DEFAULT_TASK_PROMPT = "Review the following item and provide your assessment:\n\n${item}$"


class AgenticReviewer(BaseModel):
    """An agentic literature reviewer powered by Pydantic AI.

    Wraps a Pydantic AI Agent internally. Users configure the reviewer through
    this class and never interact with pydantic_ai.Agent directly.

    When max_iterations=1 (non-agentic mode), no tools are registered and the
    agent makes a single LLM call to produce structured output — functionally
    identical to v1 reviewers.

    When max_iterations>1 (agentic mode), tools from enabled skills are
    registered and the agent can loop, call tools, build memory, flag items,
    and consult helpers before producing its final structured output.
    """

    model_config = {"arbitrary_types_allowed": True}

    # Identity
    name: str = "Reviewer"
    backstory: str = ""

    # Model — accepts Pydantic AI model string (e.g. "openai:gpt-5.4-mini") or a Model instance
    model: Union[str, Model] = "openai:gpt-5.4-mini"
    model_settings: Dict[str, Any] = Field(default_factory=dict)

    # Prompts
    system_prompt: str = ""
    task_prompt: str = DEFAULT_TASK_PROMPT

    # Output
    output_type: Type[BaseModel] = ScoringOutput

    # Agentic behavior
    max_iterations: int = 20
    agentic_effort: str = "medium"  # low, medium, high

    # Skills
    skills: List[str] = Field(default_factory=list)
    custom_skill_paths: List[Path] = Field(default_factory=list)

    # Helpers
    helpers: List["AgenticReviewer"] = Field(default_factory=list)
    helper_max_iterations: int = 5

    # Concurrency & retries
    max_concurrent_requests: int = 20
    max_retries: int = 3

    def model_post_init(self, __context: Any) -> None:
        """Validate configuration after initialization."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        if self.agentic_effort not in ("low", "medium", "high"):
            raise ValueError(f"agentic_effort must be 'low', 'medium', or 'high', got '{self.agentic_effort}'")

        if 2 <= self.max_iterations <= 4:
            warnings.warn(
                f"max_iterations={self.max_iterations} provides very limited agentic functionality. "
                f"Consider using max_iterations=1 (non-agentic) or >= 5 (full agentic).",
                UserWarning,
                stacklevel=2,
            )

    @property
    def is_agentic(self) -> bool:
        """Whether this reviewer operates in agentic mode."""
        return self.max_iterations > 1

    def _build_system_prompt(
        self,
        memory_summaries: Optional[List[Dict[str, str]]] = None,
        skill_descriptions: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Build the complete system prompt for this reviewer."""
        return build_system_prompt(
            name=self.name,
            backstory=self.backstory,
            system_prompt=self.system_prompt,
            output_type=self.output_type,
            max_iterations=self.max_iterations,
            agentic_effort=self.agentic_effort,
            enabled_skill_descriptions=skill_descriptions,
            memory_summaries=memory_summaries,
        )

    def _build_agent(
        self,
        system_prompt_str: str,
        toolsets: Optional[list] = None,
    ) -> Agent:
        """Construct the internal Pydantic AI Agent.

        Args:
            system_prompt_str: Pre-built system prompt.
            toolsets: List of FunctionToolset objects from enabled skills.

        Returns:
            Configured pydantic_ai.Agent instance.
        """
        agent_kwargs: Dict[str, Any] = {
            "deps_type": ReviewDeps,
            "system_prompt": system_prompt_str,
            "retries": self.max_retries,
            "output_type": self.output_type,
        }

        if self.model_settings:
            agent_kwargs["model_settings"] = self.model_settings

        if toolsets and self.is_agentic:
            agent_kwargs["toolsets"] = toolsets

        return Agent(self.model, **agent_kwargs)

    async def review_item(
        self,
        item_text: str,
        item_id: str = "0",
        round_id: str = "A",
        working_dir: Optional[Path] = None,
        memory_summaries: Optional[List[Dict[str, str]]] = None,
        skill_descriptions: Optional[List[Dict[str, str]]] = None,
        toolsets: Optional[list] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """Review a single item and return structured output.

        Args:
            item_text: The text content to review.
            item_id: Unique identifier for this item.
            round_id: Current review round identifier.
            working_dir: Working directory for state files.
            memory_summaries: Pre-loaded memory summaries for the system prompt.
            skill_descriptions: Enabled skill descriptions for the system prompt.
            toolsets: Pydantic AI FunctionToolset objects to register.

        Returns:
            Tuple of (response_dict, cost).
            response_dict contains the structured output fields.
            cost is the estimated cost in USD (0.0 if unavailable).
        """
        # Build system prompt
        system_prompt_str = self._build_system_prompt(
            memory_summaries=memory_summaries,
            skill_descriptions=skill_descriptions,
        )

        # Build agent
        agent = self._build_agent(system_prompt_str, toolsets)

        # Build task prompt
        user_prompt = build_task_prompt(self.task_prompt, item_text)

        # Build deps
        deps = ReviewDeps(
            item_id=item_id,
            item_text=item_text,
            agent_name=self.name,
            round_id=round_id,
            max_iterations=self.max_iterations,
            agentic_effort=self.agentic_effort,
            working_dir=working_dir,
        )

        # Set usage limits for agentic mode
        usage_limits = None
        if self.is_agentic:
            usage_limits = UsageLimits(request_limit=self.max_iterations)

        # Run the agent
        result = await agent.run(
            user_prompt,
            deps=deps,
            usage_limits=usage_limits,
        )

        # Extract structured output
        response_dict = result.output.model_dump()

        # Extract cost from usage
        cost = _estimate_cost(result.usage())

        return response_dict, cost

    async def review_items(
        self,
        text_inputs: List[str],
        item_ids: Optional[List[str]] = None,
        round_id: str = "A",
        working_dir: Optional[Path] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Review multiple items concurrently.

        Args:
            text_inputs: List of text content strings to review.
            item_ids: Optional list of item identifiers. Defaults to "0", "1", ...
            round_id: Current review round identifier.
            working_dir: Working directory for state files.

        Returns:
            Tuple of (list_of_response_dicts, total_cost).
        """
        if item_ids is None:
            item_ids = [str(i) for i in range(len(text_inputs))]

        if len(text_inputs) != len(item_ids):
            raise ValueError(f"text_inputs ({len(text_inputs)}) and item_ids ({len(item_ids)}) must have same length")

        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def _review_with_semaphore(text: str, item_id: str) -> Tuple[Dict[str, Any], float]:
            async with semaphore:
                return await self.review_item(
                    item_text=text,
                    item_id=item_id,
                    round_id=round_id,
                    working_dir=working_dir,
                )

        tasks = [_review_with_semaphore(text, iid) for text, iid in zip(text_inputs, item_ids)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses = []
        total_cost = 0.0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error reviewing item {item_ids[i]}: {result}")
                # Return None-filled response on error
                empty = {field: None for field in self.output_type.model_fields}
                empty["_error"] = str(result)
                responses.append(empty)
            else:
                resp, cost = result
                responses.append(resp)
                total_cost += cost

        return responses, total_cost


def _estimate_cost(usage: Any) -> float:
    """Estimate cost from Pydantic AI usage data.

    This is a simple estimator. Full cost tracking will be refined
    in later RFDs with provider-specific pricing lookups.

    Args:
        usage: Pydantic AI Usage object with token counts.

    Returns:
        Estimated cost in USD, or 0.0 if unavailable.
    """
    if usage is None:
        return 0.0

    # Usage object has: request_tokens, response_tokens, total_tokens
    # For now, return 0.0 — proper cost tracking in RFD-7
    return 0.0
