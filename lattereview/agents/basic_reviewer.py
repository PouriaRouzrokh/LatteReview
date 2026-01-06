"""Base agent class with consistent error handling and type safety."""

import asyncio
import datetime
import os
from pathlib import Path
from pydantic import BaseModel
import re
from typing import List, Optional, Dict, Any, Union, Callable
from tqdm.asyncio import tqdm

DEFAULT_CONCURRENT_REQUESTS = 20
DEFAULT_MAX_RETRIES = 3


class AgentError(Exception):
    """Base exception for agent-related errors."""

    pass


class BasicReviewer(BaseModel):
    generic_prompt: Optional[str] = None
    prompt_path: Optional[Union[str, Path]] = None
    response_format: Dict[str, Any] = None
    provider: Optional[Any] = None
    model_args: Dict[str, Any] = {}
    max_concurrent_requests: int = DEFAULT_CONCURRENT_REQUESTS
    name: str = "BasicReviewer"
    backstory: str = "a generic base agent"
    input_description: str = ""
    examples: Union[str, List[Union[str, Dict[str, Any]]]] = None
    reasoning: str = None
    system_prompt: Optional[str] = None
    formatted_prompt: Optional[str] = None
    cost_so_far: float = 0
    memory: List[Dict[str, Any]] = []
    identity: Dict[str, Any] = {}
    additional_context: Optional[Union[Callable, str]] = None
    verbose: bool = True
    max_retries: int = DEFAULT_MAX_RETRIES

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic model initialization."""
        try:
            self.setup()
        except Exception as e:
            raise AgentError(f"Error initializing agent: {str(e)}")

    def setup(self) -> None:
        """Build the agent's identity and configure the provider."""
        try:

            # Build the reviewer task-specific prompt
            if not self.generic_prompt and not self.prompt_path:
                raise FileNotFoundError(f"No prompt_path found. Either provide a generic_prompt or a prompt_path.")
            elif self.generic_prompt and self.prompt_path:
                self._log("Both generic_prompt and prompt_path provided. Using generic_prompt.")
            elif self.prompt_path and not self.generic_prompt:
                if not os.path.exists(self.prompt_path):
                    raise FileNotFoundError(f"Review prompt template not found at {self.prompt_path}")
                self.generic_prompt = self.prompt_path.read_text(encoding="utf-8")
            keys_to_replace = self._extract_prompt_keywords(self.generic_prompt)
            # Remove "item" and "additional_context" from the keys as they will be populated later
            keys_to_replace = [key for key in keys_to_replace if key not in ["item", "additional_context"]]
            self.formatted_prompt = self._process_prompt(
                self.generic_prompt, {key: getattr(self, key) for key in keys_to_replace}
            )

            # Build the system prompt
            self.system_prompt = self._build_system_prompt()

            # Build the agent's identity
            self.identity = {
                "system_prompt": self.system_prompt,
                "formatted_prompt": self.formatted_prompt,
                "model_args": self.model_args,
            }

            # Configure the provider
            if not self.provider:
                raise AgentError("Provider not initialized")
            self.provider.set_response_format(self.response_format)
            self.provider.system_prompt = self.system_prompt
        except Exception as e:
            raise AgentError(f"Error in setup: {str(e)}")

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        try:
            return self._clean_text(
                f"""
                Your name is: <<{self.name}>> 
                Your backstory is: <<{self.backstory}>>.
                Your task is to review input itmes with the following description: <<{self.input_description}>>.
                Your final output should have the following keys: \
                    {", ".join(f"{k} ({v})" for k, v in self.response_format.items())}.
                """
            )
        except Exception as e:
            raise AgentError(f"Error building system prompt: {str(e)}")

    def _process_prompt(self, base_prompt: str, item_dict: Dict[str, Any]) -> str:
        """Build the item prompt with variable substitution."""
        try:
            prompt = base_prompt
            if "examples" in item_dict:
                item_dict["examples"] = self._process_examples(item_dict["examples"])
            if "reasoning" in item_dict:
                item_dict["reasoning"] = self._process_reasoning(item_dict["reasoning"])

            for key, value in item_dict.items():
                if value != "" and value is not None:
                    prompt = prompt.replace(f"${{{key}}}$", str(value))
                else:
                    prompt = prompt.replace(f"${{{key}}}$", "")

            return self._clean_text(prompt)
        except Exception as e:
            raise AgentError(f"Error building item prompt: {str(e)}")

    def _process_reasoning(self, reasoning: str) -> str:
        """Process the reasoning type into a prompt string."""
        try:
            if isinstance(reasoning, str):
                reasoning = reasoning.lower()

            reasoning_map = {
                None: "",
                "brief": "Provide a brief (1-sentence) explanation for your scoring. State your reasoning before giving the score.",
                "cot": "Provide a detailed, step-by-step explanation for your scoring. State your reasoning before giving the score.",
            }
            return self._clean_text(reasoning_map.get(reasoning, ""))
        except Exception as e:
            raise AgentError(f"Error processing reasoning: {str(e)}")

    def _process_additional_context(self, context: str):
        context = f"Use the following additional context for your scoring: <<{context}>>"
        return self._clean_text(context)

    def _process_examples(self, examples: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]) -> str:
        """Process examples into a formatted string."""
        try:
            if not examples:
                return ""

            if not isinstance(examples, list):
                examples = [examples]

            examples_str = []
            for example in examples:
                if isinstance(example, dict):
                    examples_str.append("***" + "".join(f"{k}: {v}\n" for k, v in example.items()))
                elif isinstance(example, str):
                    examples_str.append("***" + example)
                else:
                    raise ValueError(f"Invalid example type: {type(example)}")

            return self._clean_text(
                "Here is one or more examples of the performance you are expected to have: \n<<"
                + "".join(examples_str)
                + ">>"
            )
        except Exception as e:
            raise AgentError(f"Error processing examples: {str(e)}")

    def reset_memory(self) -> None:
        """Reset the agent's memory and cost tracking."""
        try:
            self.memory = []
            self.cost_so_far = 0
            self.identity = {}
        except Exception as e:
            raise AgentError(f"Error resetting memory: {str(e)}")

    def _extract_prompt_keywords(self, prompt: str) -> list[str]:
        """Extracts all keywords between <<${...}$>> from the given prompt."""
        pattern = r"(?:<<)?\$\{(.*?)\}\$(?:>>)?"
        return re.findall(pattern, prompt)

    def _clean_text(self, text: str) -> str:
        """Remove extra spaces and blank lines from text."""
        try:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return " ".join(" ".join(line.split()) for line in lines)
        except Exception as e:
            raise AgentError(f"Error cleaning text: {str(e)}")

    def _log(self, x):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(x)

    async def review_items(
        self, text_input_strings: List[str], image_path_lists: List[List[str]] = None, tqdm_keywords: dict = None
    ) -> List[Dict[str, Any]]:
        """Review a list of items asynchronously with concurrency control and progress bar."""
        try:
            self.setup()
            if not image_path_lists:
                image_path_lists = [[]] * len(text_input_strings)
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)

            async def limited_review_item(
                text_input_string: str, image_path_list: List[str], index: int
            ) -> tuple[int, Dict[str, Any], Dict[str, float]]:
                async with semaphore:
                    response, input_prompt, cost = await self.review_item(text_input_string, image_path_list)
                    return index, response, input_prompt, cost

            # Building the tqdm desc
            if tqdm_keywords:
                tqdm_desc = f"""{[f'{k}: {v}' for k, v in tqdm_keywords.items()]} - \
                    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
            else:
                tqdm_desc = f"Reviewing {len(text_input_strings)} items - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            # Create tasks with indices
            tasks = [
                limited_review_item(text_input_string, image_path_list, i)
                for i, (text_input_string, image_path_list) in enumerate(zip(text_input_strings, image_path_lists))
            ]

            # Collect results with indices
            initial_results = []
            async for result in tqdm(asyncio.as_completed(tasks), total=len(text_input_strings), desc=tqdm_desc):
                initial_results.append(await result)

            # Sort by original index and separate response and cost
            initial_results.sort(key=lambda x: x[0])  # Sort by index
            results = []

            for i, response, input_prompt, cost in initial_results:
                if isinstance(cost, dict):
                    cost = cost["total_cost"]
                self.cost_so_far += cost
                results.append(response)
                self.memory.append(
                    {
                        "system_prompt": self.system_prompt,
                        "model_args": self.model_args,
                        "input_prompt": input_prompt,
                        "response": response,
                        "cost": cost,
                    }
                )

            return results, cost
        except Exception as e:
            raise AgentError(f"Error reviewing items: {str(e)}")

    async def review_item(
        self, text_input_string: str, image_path_list: List[str] = []
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Review a single item asynchronously with error handling."""
        num_tried = 0
        while num_tried < self.max_retries:
            try:
                input_prompt = self._process_prompt(self.formatted_prompt, {"item": text_input_string})
                if self.additional_context == "" or not self.additional_context:
                    context = self.additional_context
                elif isinstance(self.additional_context, str):
                    context = self._process_additional_context(self.additional_context)
                elif isinstance(self.additional_context, Callable):
                    context = await self.additional_context(text_input_string)
                    context = self._process_additional_context(context)
                else:
                    raise AgentError("Additional context must be a string or callable")
                input_prompt = self._process_prompt(input_prompt, {"additional_context": context})
                response, cost = await self.provider.get_json_response(input_prompt, image_path_list, **self.model_args)
                return response, input_prompt, cost
            except Exception as e:
                num_tried += 1
                self._log(f"Error reviewing item: {str(e)}. Retrying {num_tried}/{self.max_retries}")
        raise AgentError("Error reviewing item!")
