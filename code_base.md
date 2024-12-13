## collect_scripts.py

```py
"""Script to collect and organize code files into a markdown document."""
from pathlib import Path
from typing import List, Tuple, Set

def gather_code_files(
    root_dir: Path,
    extensions: Set[str],
    exclude_files: Set[str],
    exclude_folders: Set[str]
) -> Tuple[List[Path], List[Path]]:
    """Gather code files while respecting exclusion rules."""
    try:
        code_files: List[Path] = []
        excluded_files_found: List[Path] = []
        
        for file_path in root_dir.rglob('*'):
            if any(excluded in file_path.parts for excluded in exclude_folders):
                if file_path.is_file():
                    excluded_files_found.append(file_path)
                continue
            
            if file_path.is_file():
                if file_path.name in exclude_files:
                    excluded_files_found.append(file_path)
                elif file_path.suffix in extensions:
                    code_files.append(file_path)
        
        return code_files, excluded_files_found
    except Exception as e:
        raise RuntimeError(f"Error gathering code files: {str(e)}")

def write_to_markdown(
    code_files: List[Path],
    excluded_files: List[Path],
    output_file: Path
) -> None:
    """Write collected files to a markdown document."""
    try:
        with output_file.open('w', encoding='utf-8') as md_file:
            for file_path in code_files:
                relative_path = file_path.relative_to(file_path.cwd())
                md_file.write(f"## {relative_path}\n\n")
                md_file.write("```" + file_path.suffix.lstrip('.') + "\n")
                md_file.write(file_path.read_text(encoding='utf-8'))
                md_file.write("\n```\n\n")
    except Exception as e:
        raise RuntimeError(f"Error writing markdown file: {str(e)}")

def create_markdown(
    root_dir: Path,
    extensions: Set[str],
    exclude_files: Set[str],
    exclude_folders: Set[str],
    output_file: Path = Path('code_base.md')
) -> None:
    """Create a markdown file containing all code files."""
    try:
        code_files, excluded_files = gather_code_files(root_dir, extensions, exclude_files, exclude_folders)
        write_to_markdown(code_files, excluded_files, output_file)
        print(f"Markdown file '{output_file}' created with {len(code_files)} code files and {len(excluded_files)} excluded files.")
    except Exception as e:
        raise RuntimeError(f"Error creating markdown: {str(e)}")

if __name__ == "__main__":
    root_directory = Path(__file__).parent
    extensions_to_look_for = {'.py', '.txt'}
    exclude_files_list = {'personal_apis.js'}
    exclude_folders_list = {'venv'}
    
    create_markdown(root_directory, extensions_to_look_for, exclude_files_list, exclude_folders_list)
```

## test/__init__.py

```py

```

## lattereview/__init__.py

```py

```

## lattereview/review.py

```py

```

## lattereview/generic_prompts/review_prompt.txt

```txt
Review the item below and evaluate it against the following criteria:

Input item: <<${item}$>>

Review criteria: <<${review_criteria}$>>

Now score the article based on your evaluation and by choosing one of the following values: ${score_set}$.

If you are highly in doubt about what to score, score "0". 

Your scoring should be based on the following rules: <<${scoring_rules}$>>

${reasoning}$

${examples}$



```

## lattereview/generic_prompts/__init__.py

```py

```

## lattereview/providers/base_provider.py

```py
"""Base class for all API providers with consistent error handling and type hints."""
from typing import Optional, Any, List, Dict, Union
import pydantic
from tokencost import calculate_prompt_cost, calculate_completion_cost

class ProviderError(Exception):
    """Base exception for provider-related errors."""
    pass

class ClientCreationError(ProviderError):
    """Raised when client creation fails."""
    pass

class ResponseError(ProviderError):
    """Raised when getting a response fails."""
    pass

class BaseProvider(pydantic.BaseModel):
    provider: str = "DefaultProvider"
    client: Optional[Any] = None
    api_key: Optional[str] = None
    model: str = "default-model"
    system_prompt: str = "You are a helpful assistant."
    response_format: Optional[Dict[str, Any]] = None
    last_response: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def create_client(self) -> Any:
        """Create and initialize the client for the provider."""
        raise NotImplementedError("Subclasses must implement create_client")
    
    def set_response_format(self, response_format: Dict[str, Any]) -> None:
        """Set the response format for the provider."""
        raise NotImplementedError("Subclasses must implement set_response_format")

    async def get_response(
        self, 
        messages: Union[str, List[str]], 
        message_list: Optional[List[Dict[str, str]]] = None, 
        system_message: Optional[str] = None
    ) -> tuple[Any, Dict[str, float]]:
        """Get a response from the provider."""
        raise NotImplementedError("Subclasses must implement get_response")

    async def get_json_response(
        self, 
        messages: Union[str, List[str]], 
        message_list: Optional[List[Dict[str, str]]] = None, 
        system_message: Optional[str] = None
    ) -> tuple[Any, Dict[str, float]]:
        """Get a JSON-formatted response from the provider."""
        raise NotImplementedError("Subclasses must implement get_json_response")

    def _prepare_message_list(
        self, 
        message: str, 
        message_list: Optional[List[Dict[str, str]]] = None, 
        system_message: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Prepare the list of messages to be sent to the provider."""
        raise NotImplementedError("Subclasses must implement _prepare_message_list")

    async def _fetch_response(
        self, 
        message_list: List[Dict[str, str]], 
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fetch the raw response from the provider."""
        raise NotImplementedError("Subclasses must implement _fetch_response")

    async def _fetch_json_response(
        self, 
        message_list: List[Dict[str, str]], 
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fetch the JSON-formatted response from the provider."""
        raise NotImplementedError("Subclasses must implement _fetch_json_response")

    def _extract_content(self, response: Any) -> Any:
        """Extract content from the provider's response."""
        raise NotImplementedError("Subclasses must implement _extract_content")
    
    def _get_cost(self, input_messages: List[str], completion_text: str) -> Dict[str, float]:
        """Calculate the cost of a prompt completion."""
        try:
            print(input_messages, self.model)
            input_cost = calculate_prompt_cost(input_messages, self.model)
            output_cost = calculate_completion_cost(completion_text, self.model)
            return {
                "input_cost": float(input_cost),
                "output_cost": float(output_cost),
                "total_cost": float(input_cost + output_cost)
            }
        except Exception as e:
            raise ProviderError(f"Error calculating costs: {str(e)}")
```

## lattereview/providers/__init__.py

```py

```

## lattereview/providers/openai_provider.py

```py
"""OpenAI API provider implementation with comprehensive error handling and type safety."""
from typing import Optional, List, Dict, Any, Union, Tuple
import os
from pydantic import BaseModel, create_model
import openai
from .base_provider import BaseProvider, ProviderError, ClientCreationError, ResponseError

class OpenAIProvider(BaseProvider):
    provider: str = "OpenAI"
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    client: Optional[openai.AsyncOpenAI] = None
    model: str = "gpt-4o-mini"
    response_format_class: Optional[BaseModel] = None

    def __init__(self, **data: Any) -> None:
        """Initialize the OpenAI provider with error handling."""
        super().__init__(**data)
        try:
            self.client = self.create_client()
        except Exception as e:
            raise ClientCreationError(f"Failed to create OpenAI client: {str(e)}")

    def set_response_format(self, response_format: Dict[str, Any]) -> None:
        """Set the response format for JSON responses."""
        try:
            if not isinstance(response_format, dict):
                raise ValueError("Response format must be a dictionary")
            self.response_format = response_format
            fields = {key: (value, ...) for key, value in response_format.items()}
            self.response_format_class = create_model('ResponseFormat', **fields)
        except Exception as e:
            raise ProviderError(f"Error setting response format: {str(e)}")

    def create_client(self) -> openai.AsyncOpenAI:
        """Create and return the OpenAI client."""
        if not self.api_key:
            raise ClientCreationError("OPENAI_API_KEY environment variable is not set")
        try:
            return openai.AsyncOpenAI(api_key=self.api_key)
        except Exception as e:
            raise ClientCreationError(f"Failed to create OpenAI client: {str(e)}")

    async def get_response(
        self,
        messages: str,
        message_list: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any
    ) -> Tuple[Any, Dict[str, float]]:
        """Get a response from OpenAI."""
        try:
            message_list = self._prepare_message_list(messages, message_list)
            response = await self._fetch_response(message_list, kwargs)
            txt_response = self._extract_content(response)
            cost = self._get_cost(input_messages=messages, completion_text=txt_response)
            return txt_response, cost
        except Exception as e:
            raise ResponseError(f"Error getting response: {str(e)}")

    async def get_json_response(
        self,
        messages: str,
        message_list: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any
    ) -> Tuple[Any, Dict[str, float]]:
        """Get a JSON response from OpenAI."""
        try:
            if not self.response_format_class:
                raise ValueError("Response format is not set")
            message_list = self._prepare_message_list(messages, message_list)
            response = await self._fetch_json_response(message_list, kwargs)
            txt_response = self._extract_content(response)
            cost = self._get_cost(input_messages=messages, completion_text=txt_response)
            return txt_response, cost
        except Exception as e:
            raise ResponseError(f"Error getting JSON response: {str(e)}")

    def _prepare_message_list(
        self,
        message: str,
        message_list: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Prepare the message list for the API call."""
        try:
            if message_list:
                message_list.append({"role": "user", "content": message})
            else:
                message_list = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message},
                ]
            return message_list
        except Exception as e:
            raise ProviderError(f"Error preparing message list: {str(e)}")

    async def _fetch_response(
        self,
        message_list: List[Dict[str, str]],
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fetch the raw response from OpenAI."""
        try:
            return await self.client.chat.completions.create(
                model=self.model,
                messages=message_list,
                **(kwargs or {})
            )
        except Exception as e:
            raise ResponseError(f"Error fetching response: {str(e)}")

    async def _fetch_json_response(
        self,
        message_list: List[Dict[str, str]],
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fetch the JSON response from OpenAI."""
        try:
            return await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=message_list,
                response_format=self.response_format_class,
                **(kwargs or {})
            )
        except Exception as e:
            raise ResponseError(f"Error fetching JSON response: {str(e)}")

    def _extract_content(self, response: Any) -> str:
        """Extract content from the response."""
        try:
            if not response:
                raise ValueError("Empty response received")
            self.last_response = response
            return response.choices[0].message.content
        except Exception as e:
            raise ResponseError(f"Error extracting content: {str(e)}")
```

## lattereview/agents/base_agent.py

```py
"""Base agent class with consistent error handling and type safety."""
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel

class ReasoningType(Enum):
    """Enumeration for reasoning types."""
    NONE = "none"
    BRIEF = "brief"
    LONG = "long"
    COT = "cot"

class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass

class BaseAgent(BaseModel):
    response_format: Dict[str, Any]
    provider: Optional[Any] = None
    max_concurrent_requests: int = 20
    name: str = "BaseAgent"
    backstory: str = "a generic base agent"
    input_description: str = "article title/abstract"
    examples: Union[str, List[Union[str, Dict[str, Any]]]] = None
    reasoning: ReasoningType = ReasoningType.BRIEF
    system_prompt: Optional[str] = None
    item_prompt: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 150
    cost_so_far: float = 0
    memory: List[Dict[str, Any]] = []
    identity: Dict[str, Any] = {}

    def __init__(self, **data: Any) -> None:
        """Initialize the base agent with error handling."""
        try:
            super().__init__(**data)
            if isinstance(self.reasoning, str):
                self.reasoning = ReasoningType(self.reasoning.lower())
            if self.reasoning == ReasoningType.NONE:
                self.response_format.pop("reasoning", None)
            self.setup()
        except Exception as e:
            raise AgentError(f"Error initializing agent: {str(e)}")

    def setup(self) -> None:
        """Setup the agent before use."""
        raise NotImplementedError("This method must be implemented by subclasses.")

    def build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        try:
            return self._clean_text(f"""
                Your name is <<{self.name}>> and you are <<{self.backstory}>>.
                Your task is to review input itmes with the following description: <<{self.input_description}>>.
                Your final output should have the following keys: 
                {", ".join(f"{k} ({v})" for k, v in self.response_format.items())}.
                """)
        except Exception as e:
            raise AgentError(f"Error building system prompt: {str(e)}")
    
    def build_item_prompt(self, base_prompt: str, item_dict: Dict[str, Any]) -> str:
        """Build the item prompt with variable substitution."""
        try:
            prompt = base_prompt
            if "examples" in item_dict:
                item_dict["examples"] = self.process_examples(item_dict["examples"])
            if "reasoning" in item_dict:
                item_dict["reasoning"] = self.process_reasoning(item_dict["reasoning"])
            
            for key, value in item_dict.items():
                if value is not None:
                    prompt = prompt.replace(f'${{{key}}}$', str(value))
                else:
                    prompt = prompt.replace(f'${{{key}}}$', '')
            
            return self._clean_text(prompt)
        except Exception as e:
            raise AgentError(f"Error building item prompt: {str(e)}")
    
    def process_reasoning(self, reasoning: Union[str, ReasoningType]) -> str:
        """Process the reasoning type into a prompt string."""
        try:
            if isinstance(reasoning, str):
                reasoning = ReasoningType(reasoning.lower())
            
            reasoning_map = {
                ReasoningType.NONE: "",
                ReasoningType.BRIEF: "You must also provide a brief (1 sentence) reasoning for your scoring. First reason then score!",
                ReasoningType.LONG: "You must also provide a detailed reasoning for your scoring. First reason then score!",
                ReasoningType.COT: "You must also provide a reasoning for your scoring . Think step by step in your reasoning. First reason then score!"
            }
            
            return self._clean_text(reasoning_map.get(reasoning, ""))
        except Exception as e:
            raise AgentError(f"Error processing reasoning: {str(e)}")
    
    def process_examples(self, examples: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]) -> str:
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
            
            return self._clean_text("<<Here is one or more examples of the performance you are expected to have: \n" + 
                                  "".join(examples_str)+">>")
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

    def _clean_text(self, text: str) -> str:
        """Remove extra spaces and blank lines from text."""
        try:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return ' '.join(' '.join(line.split()) for line in lines)
        except Exception as e:
            raise AgentError(f"Error cleaning text: {str(e)}")

    async def review_items(self, items: List[str]) -> List[Dict[str, Any]]:
        """Review a list of items asynchronously."""
        raise NotImplementedError("This method must be implemented by subclasses.")

    async def review_item(self, item: str) -> Dict[str, Any]:
        """Review a single item asynchronously."""
        raise NotImplementedError("This method must be implemented by subclasses.")
```

## lattereview/agents/__init__.py

```py

```

## lattereview/agents/scoring_reviewer.py

```py
"""Reviewer agent implementation with consistent error handling and type safety."""
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path
from pydantic import Field
from .base_agent import BaseAgent, AgentError

class ScoringReviewer(BaseAgent):
    response_format: Dict[str, Any] = {
        "score": int,
        "reasoning": str,
    }
    review_criteria: Optional[str] = None
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
            raise AgentError(f"Error in post-initialization: {str(e)}")

    def setup(self) -> None:
        """Build the agent's identity and configure the provider."""
        try:
            self.system_prompt = self.build_system_prompt()
            self.score_set = str(self.score_set)
            keys_to_replace = ['review_criteria', 'score_set', 
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
                self.memory.append({
                    'identity': self.identity,
                    'item': item,
                    'response': response,
                    'cost': cost
                })
                results.append(response)

            return results
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
```

