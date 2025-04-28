"""Ollama API provider implementation using AsyncClient with comprehensive error handling and type safety."""

import inspect
from typing import Optional, List, Dict, Any, Union, Tuple, AsyncGenerator
import json
from ollama import AsyncClient
from pydantic import BaseModel, create_model
from .base_provider import BaseProvider, ProviderError, ClientCreationError, ResponseError, InvalidResponseFormatError


class OllamaProvider(BaseProvider):
    provider: str = "Ollama"
    client: Optional[AsyncClient] = None
    model: str = "llama3.2-vision:latest"  # Default model
    response_format_class: Optional[Any] = None
    invalid_keywords: List[str] = ["temperature", "max_tokens"]
    host: str = "http://localhost:11434"  # Default Ollama API endpoint

    def __init__(self, **data: Any) -> None:
        """Initialize the Ollama provider with error handling."""
        super().__init__(**data)
        try:
            self.client = self.create_client()
        except Exception as e:
            raise ClientCreationError(f"Failed to initialize Ollama: {str(e)}")

    def _clean_kwargs(self, kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Remove invalid keywords from kwargs."""
        if kwargs is None:
            return {}

        cleaned_kwargs = kwargs.copy()
        for keyword in self.invalid_keywords:
            cleaned_kwargs.pop(keyword, None)
        return cleaned_kwargs

    def set_response_format(self, response_format: Dict[str, Any]) -> None:
        """Set the response format for JSON responses."""
        try:
            if not response_format:
                raise InvalidResponseFormatError("Response format cannot be empty")
            if isinstance(response_format, dict):
                self.response_format = response_format
                fields = {key: (value, ...) for key, value in response_format.items()}
                self.response_format_class = create_model("ResponseFormat", **fields)
            elif self._check_basemodel_class(response_format):
                self.response_format_class = response_format
        except Exception as e:
            raise ProviderError(f"Error setting response format: {str(e)}")

    def create_client(self) -> AsyncClient:
        """Create and return the Ollama AsyncClient."""
        try:
            return AsyncClient(host=self.host)
        except Exception as e:
            raise ClientCreationError(f"Failed to create Ollama client: {str(e)}")

    async def get_response(
        self,
        input_prompt: str,
        image_path_list: List[str] = [],
        message_list: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[Tuple[Any, Dict[str, float]], AsyncGenerator[str, None]]:
        """Get a response from Ollama, with optional streaming support."""
        try:
            message_list = self._prepare_message_list(input_prompt, image_path_list, message_list)

            if stream:
                return self._stream_response(message_list, kwargs)
            else:
                response = await self._fetch_response(message_list, kwargs)
                txt_response = self._extract_content(response)
                cost = {
                    "input_cost": 0,
                    "output_cost": 0,
                    "total_cost": 0,
                }  # Ollama models are local and therefore free.
                return txt_response, cost
        except Exception as e:
            raise ResponseError(f"Error getting response: {str(e)}")

    async def get_json_response(
        self,
        input_prompt: str,
        image_path_list: List[str] = [],
        message_list: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, float]]:
        """Get a JSON response from Ollama using the defined schema."""
        try:
            if not self.response_format_class:
                raise ValueError("Response format is not set")

            message_list = self._prepare_message_list(input_prompt, image_path_list, message_list)

            # Attempting fix for ollama not supporting response_format - To be removed if the fix is confirmed
            # Update system message to request JSON output
            # if message_list and message_list[0]["role"] == "system":
            #     schema_str = json.dumps(self.response_format_class.model_json_schema(), indent=2)
            #     message_list[0]["content"] = (
            #         f"{message_list[0]['content']}\n\n"
            #         f"Please provide your response as a JSON object following this schema:\n{schema_str}"
            #     )

            # Set format parameter to 'json'
            cleaned_kwargs = self._clean_kwargs(kwargs)
            cleaned_kwargs["format"] = "json"

            response = await self._fetch_response(message_list, cleaned_kwargs)
            txt_response = self._extract_content(response)
            cost = {"input_cost": 0, "output_cost": 0, "total_cost": 0}  # Ollama models are local and therefore free.
            return txt_response, cost
        except Exception as e:
            raise ResponseError(f"Error getting JSON response: {str(e)}")

    def _prepare_message_list(
        self,
        input_prompt: str,
        image_path_list: List[str],
        message_list: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Prepare the message list for the API call."""
        try:
            if len(image_path_list) == 0:
                image_path_list = None
            if message_list:
                message_list.append({"role": "user", "content": input_prompt, "images": image_path_list})
            else:
                message_list = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": input_prompt, "images": image_path_list},
                ]
            return message_list
        except Exception as e:
            raise ProviderError(f"Error preparing message list: {str(e)}")

    async def _fetch_response(self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """Fetch the raw response from Ollama using AsyncClient."""
        try:
            if not self.client:
                raise ValueError("Client not initialized")

            cleaned_kwargs = self._clean_kwargs(kwargs)
            response = await self.client.chat(model=self.model, messages=message_list, **cleaned_kwargs)
            return response
        except Exception as e:
            raise ResponseError(f"Error fetching response: {str(e)}")

    async def _stream_response(
        self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream the response from Ollama."""
        try:
            if not self.client:
                raise ValueError("Client not initialized")

            cleaned_kwargs = self._clean_kwargs(kwargs)
            cleaned_kwargs["stream"] = True

            async for part in self.client.chat(model=self.model, messages=message_list, **cleaned_kwargs):
                yield part.message.content
        except Exception as e:
            raise ResponseError(f"Error streaming response: {str(e)}")

    def _extract_content(self, response: Any) -> str:
        """Extract content from the response."""
        try:
            if not response:
                raise ValueError("Empty response received")
            self.last_response = response
            return response.message.content
        except Exception as e:
            raise ResponseError(f"Error extracting content: {str(e)}")

    async def close(self) -> None:
        """Close the client session."""
        if self.client:
            await self.client.aclose()

    def _check_basemodel_class(self, arg):
        """Check if the argument is a Pydantic BaseModel class."""
        return inspect.isclass(arg) and issubclass(arg, BaseModel)
