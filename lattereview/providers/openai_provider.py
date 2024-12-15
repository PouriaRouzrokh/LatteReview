"""OpenAI API provider implementation with comprehensive error handling and type safety."""

from typing import Optional, List, Dict, Any, Tuple
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
            self.response_format_class = create_model("ResponseFormat", **fields)
        except Exception as e:
            raise ProviderError(f"Error setting response format: {str(e)}")

    def create_client(self) -> openai.AsyncOpenAI:
        """Create and return the OpenAI client."""
        gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        if not self.api_key:
            raise ClientCreationError("OPENAI_API_KEY environment variable is not set")
        try:
            if "gemini" not in self.model.lower():
                return openai.AsyncOpenAI(api_key=self.api_key)
            self.api_key = os.getenv("GEMINI_API_KEY", self.api_key)
            return openai.AsyncOpenAI(api_key=self.api_key, base_url=gemini_base_url)
        except Exception as e:
            raise ClientCreationError(f"Failed to create OpenAI client: {str(e)}")

    async def get_response(
        self, messages: str, message_list: Optional[List[Dict[str, str]]] = None, **kwargs: Any
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
        self, messages: str, message_list: Optional[List[Dict[str, str]]] = None, **kwargs: Any
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
        messages: str,
        message_list: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Prepare the message list for the API call."""
        try:
            if message_list:
                message_list.append({"role": "user", "content": messages})
            else:
                message_list = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": messages},
                ]
            return message_list
        except Exception as e:
            raise ProviderError(f"Error preparing message list: {str(e)}")

    async def _fetch_response(self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """Fetch the raw response from OpenAI."""
        try:
            return await self.client.chat.completions.create(model=self.model, messages=message_list, **(kwargs or {}))
        except Exception as e:
            raise ResponseError(f"Error fetching response: {str(e)}")

    async def _fetch_json_response(
        self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fetch the JSON response from OpenAI."""
        try:
            return await self.client.beta.chat.completions.parse(
                model=self.model, messages=message_list, response_format=self.response_format_class, **(kwargs or {})
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
