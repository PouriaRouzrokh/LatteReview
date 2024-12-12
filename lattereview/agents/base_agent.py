"""Base agent class with consistent error handling and type safety."""
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
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
                Your name is {self.name} and you are {self.backstory}
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
                ReasoningType.BRIEF: "Please provide a brief reasoning for your response.",
                ReasoningType.LONG: "Please provide a detailed reasoning for your response.",
                ReasoningType.COT: "Please provide a reasoning for your response. Think step by step for the best explanation."
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
            
            return self._clean_text("Here is one or more examples of the performance you are expected to have: \n" + 
                                  "".join(examples_str))
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