from typing import List, Dict, Any, Optional
import asyncio
from pydantic import BaseModel

from .base_agent import BaseAgent

class ReviewerResponseFormat(BaseModel):
    reasoning: str
    score: int
    question: str

class ReviewerAgent(BaseAgent):
    response_format: Any = ReviewerResponseFormat
    review_type: str = "title/abstract"
    review_criteria: Optional[str] = None
    scoring_schema: str = "[0, 1]"
    scoring_rules: str = "Your scores should follow the defined schema."
    question_criteria: str = "When highly uncertain about a review."
    review_prompt: str = "Default review prompt."

    def __init__(self, **data):
        """Initialize the ReviewerAgent."""
        super().__init__(**data) 
        self.review_prompt = open('../lattereview/prompts/review_prompt.txt', 'r').read()
        self.build_identity()

    def build_identity(self) -> None:
        """Build the agent's identity based on its attributes. """
        self.identity = {
            'name': self.name,
            'backstory': self.backstory,
            'temperature': self.temperature,
            'review_type': self.review_type,
            'review_criteria': self.review_criteria,
            'scoring_schema': self.scoring_schema,
            'scoring_rules': self.scoring_rules,
            'question_criteria': self.question_criteria,
            'system_prompt': f"Your name is {self.name} and you are {self.backstory}"
        }

    async def review_items(self, items: List[str]) -> List[Dict]:
        """Review a list of items asynchronously."""
        self.build_identity()
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def limited_review_item(item):
            async with semaphore:  # Limit concurrency
                return await self.review_item(item)

        responses_costs = await asyncio.gather(*(limited_review_item(item) for item in items))
        responses, costs = zip(*responses_costs)
        for item, response, cost in zip(items, responses, costs):
            self.memory.append(
                {
                    'identity': self.identity,
                    'item': item,
                    'response': response,
                    'cost': cost
                }
            )
        return responses

    async def review_item(self, item: str) -> Dict:
        """Review a single item asynchronously."""
        keys_to_replace = ['review_type', 'review_criteria', 'scoring_schema', 'scoring_rules', 'question_criteria']
        input_prompt = self.complete_prompt(self.review_prompt, {key: self.identity[key] for key in keys_to_replace})
        input_prompt = self.complete_prompt(input_prompt, {'item': item})
        response, cost = await self.provider.get_json_response(input_prompt, temperature=self.temperature)
        return response, cost