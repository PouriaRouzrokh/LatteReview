"""Reviewer agent implementation with consistent error handling and type safety."""
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import Field
from .base_agent import BaseAgent, AgentError

class ScoringReviewer(BaseAgent):
    response_format: Dict[str, Any] = {
        "reasoning": str,
        "score": int,
    }
    scoring_task: Optional[str] = None
    score_set: List[int] = [1, 2]
    scoring_rules: str = "Your scores should follow the defined schema."
    generic_item_prompt: Optional[str] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic model initialization."""
        try:
            assert 0 not in self.score_set, "Score set must not contain 0. This value is reserved for uncertain scorings / errors."
            prompt_path = Path(__file__).parent.parent / "generic_prompts" / "review_prompt.txt"
            if not prompt_path.exists():
                raise FileNotFoundError(f"Review prompt template not found at {prompt_path}")
            self.generic_item_prompt = prompt_path.read_text(encoding='utf-8')
            self.setup()
        except Exception as e:
            raise AgentError(f"Error initalizing agent: {str(e)}")

    def setup(self) -> None:
        """Build the agent's identity and configure the provider."""
        try:
            self.system_prompt = self.build_system_prompt()
            self.score_set = str(self.score_set)
            keys_to_replace = ['scoring_task', 'score_set', 
                             'scoring_rules', 'reasoning', 'examples']
            
            self.item_prompt = self.build_item_prompt(
                self.generic_item_prompt,
                {key: getattr(self, key) for key in keys_to_replace}
            )
            
            self.identity = {
                "system_prompt": self.system_prompt,
                "item_prompt": self.item_prompt,
                'temperature': self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            if not self.provider:
                raise AgentError("Provider not initialized")
            
            self.provider.set_response_format(self.response_format)
            self.provider.system_prompt = self.system_prompt
        except Exception as e:
            raise AgentError(f"Error in setup: {str(e)}")

    async def review_items(self, items: List[str]) -> List[Dict[str, Any]]:
        """Review a list of items asynchronously with concurrency control."""
        try:
            self.setup()
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)

            async def limited_review_item(item: str) -> tuple[Dict[str, Any], Dict[str, float]]:
                async with semaphore:
                    return await self.review_item(item)

            responses_costs = await asyncio.gather(
                *(limited_review_item(item) for item in items),
                return_exceptions=True
            )

            # Handle any exceptions from the gathered tasks
            results = []
            for item, response_cost in zip(items, responses_costs):
                if isinstance(response_cost, Exception):
                    print(f"Error processing item: {item}, Error: {str(response_cost)}")
                    continue
                
                response, cost = response_cost
                self.cost_so_far += cost["total_cost"]
                results.append(response)
                self.memory.append({
                    'identity': self.identity,
                    'item': item,
                    'response': response,
                    'cost': cost
                })

            return results, cost["total_cost"]
        except Exception as e:
            raise AgentError(f"Error reviewing items: {str(e)}")

    async def review_item(self, item: str) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Review a single item asynchronously with error handling."""
        try:
            item_prompt = self.build_item_prompt(self.item_prompt, {'item': item})
            response, cost = await self.provider.get_json_response(
                item_prompt,
                temperature=self.temperature
            )
            return response, cost
        except Exception as e:
            raise AgentError(f"Error reviewing item: {str(e)}")