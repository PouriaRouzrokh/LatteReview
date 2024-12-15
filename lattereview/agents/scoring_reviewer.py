"""Reviewer agent implementation with consistent error handling and type safety."""

import asyncio
from pathlib import Path
import datetime
from typing import List, Dict, Any, Optional
from pydantic import Field
from .base_agent import BaseAgent, AgentError, ReasoningType
from tqdm.asyncio import tqdm
import warnings

DEFAULT_MAX_RETRIES = 3


class ScoringReviewer(BaseAgent):
    response_format: Dict[str, Any] = {
        "reasoning": str,
        "score": int,
    }
    scoring_task: Optional[str] = None
    score_set: List[int] = [1, 2]
    scoring_rules: str = "Your scores should follow the defined schema."
    generic_item_prompt: Optional[str] = Field(default=None)
    reasoning: ReasoningType = ReasoningType.BRIEF
    max_retries: int = DEFAULT_MAX_RETRIES

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic model initialization."""
        try:
            assert self.reasoning != ReasoningType.NONE, "Reasoning type cannot be 'none' for ScoreReviewer"
            assert (
                0 not in self.score_set
            ), "Score set must not contain 0. This value is reserved for uncertain scorings / errors."
            prompt_path = Path(__file__).parent.parent / "generic_prompts" / "review_prompt.txt"
            if not prompt_path.exists():
                raise FileNotFoundError(f"Review prompt template not found at {prompt_path}")
            self.generic_item_prompt = prompt_path.read_text(encoding="utf-8")
            self.setup()
        except Exception as e:
            raise AgentError(f"Error initializing agent: {str(e)}")

    def setup(self) -> None:
        """Build the agent's identity and configure the provider."""
        try:
            self.system_prompt = self.build_system_prompt()
            self.score_set = str(self.score_set)
            keys_to_replace = ["scoring_task", "score_set", "scoring_rules", "reasoning", "examples"]

            self.item_prompt = self.build_item_prompt(
                self.generic_item_prompt, {key: getattr(self, key) for key in keys_to_replace}
            )

            self.identity = {
                "system_prompt": self.system_prompt,
                "item_prompt": self.item_prompt,
                "model_args": self.model_args,
            }

            if not self.provider:
                raise AgentError("Provider not initialized")

            self.provider.set_response_format(self.response_format)
            self.provider.system_prompt = self.system_prompt
        except Exception as e:
            raise AgentError(f"Error in setup: {str(e)}")

    async def review_items(self, items: List[str], tqdm_keywords: dict = None) -> List[Dict[str, Any]]:
        """Review a list of items asynchronously with concurrency control and progress bar."""
        try:
            self.setup()
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)

            async def limited_review_item(item: str, index: int) -> tuple[int, Dict[str, Any], Dict[str, float]]:
                async with semaphore:
                    response, cost = await self.review_item(item)
                    return index, response, cost

            # Building the tqdm desc
            if tqdm_keywords:
                tqdm_desc = f"""{[f'{k}: {v}' for k, v in tqdm_keywords.items()]} - \
                    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
            else:
                tqdm_desc = f"Reviewing {len(items)} items - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            # Create tasks with indices
            tasks = [limited_review_item(item, i) for i, item in enumerate(items)]

            # Collect results with indices
            responses_costs = []
            async for result in tqdm(asyncio.as_completed(tasks), total=len(items), desc=tqdm_desc):
                responses_costs.append(await result)

            # Sort by original index and separate response and cost
            responses_costs.sort(key=lambda x: x[0])  # Sort by index
            results = []

            for i, response, cost in responses_costs:
                if isinstance(cost, dict):
                    cost = cost["total_cost"]
                self.cost_so_far += cost
                results.append(response)
                self.memory.append(
                    {
                        "identity": self.identity,
                        "item": items[i],
                        "response": response,
                        "cost": cost,
                        "model_args": self.model_args,
                    }
                )

            return results, cost
        except Exception as e:
            raise AgentError(f"Error reviewing items: {str(e)}")

    async def review_item(self, item: str) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Review a single item asynchronously with error handling."""
        num_tried = 0
        while num_tried < self.max_retries:
            try:
                item_prompt = self.build_item_prompt(self.item_prompt, {"item": item})
                response, cost = await self.provider.get_json_response(item_prompt, **self.model_args)
                return response, cost
            except Exception as e:
                warnings.warn(f"Error reviewing item: {str(e)}. Retrying {num_tried}/{self.max_retries}")
        raise AgentError("Error reviewing item!")
