"""OpenAI API provider implementation with comprehensive error handling and type safety."""

import base64
import inspect
from typing import Optional, List, Dict, Any, Tuple, Union
import os
from pydantic import BaseModel, create_model
import openai
from .base_provider import BaseProvider, ProviderError, ClientCreationError, ResponseError, InvalidResponseFormatError


class OpenAIProvider(BaseProvider):
    provider: str = "OpenAI"
    api_key: str = None
    base_url: str = None
    model: str = "gpt-6-luna"
    response_format_class: Optional[Any] = None

    def __init__(self, **data: Any) -> None:
        """Initialize the OpenAI provider with error handling."""
        super().__init__(**data)
        try:
            self.client = self.create_client(base_url=self.base_url)
        except Exception as e:
            raise ClientCreationError(f"Failed to create OpenAI client: {str(e)}")

    def set_response_format(self, response_format: Union[BaseModel, Dict[str, Any]]) -> None:
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

    def create_client(self, base_url: Optional[str] = None) -> openai.AsyncOpenAI:
        """Create and return the OpenAI client."""
        gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        openai_base_url = "https://api.openai.com/v1"
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                self.api_key = os.getenv("GEMINI_API_KEY")
                if not self.api_key:
                    raise ClientCreationError(
                        "OPENAI_API_KEY or GEMINI_API_KEY environment variable is not set. Please pass your API key or set this variable."
                    )

        if "gemini" in self.model.lower():
            self.api_key = os.getenv("GEMINI_API_KEY", self.api_key)
            base_url = base_url or gemini_base_url
            try:
                return openai.AsyncOpenAI(api_key=self.api_key, base_url=base_url)
            except Exception as e:
                raise ClientCreationError(f"Failed to create OpenAI client: {str(e)}")
        else:
            base_url = base_url or openai_base_url
            try:
                return openai.AsyncOpenAI(api_key=self.api_key, base_url=base_url)
            except Exception as e:
                raise ClientCreationError(f"Failed to create OpenAI client: {str(e)}")

    async def get_response(
        self,
        input_prompt: str,
        image_path_list: List[str] = [],
        message_list: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, float]]:
        """Get a response from OpenAI."""
        try:
            message_list = self._prepare_message_list(input_prompt, image_path_list, message_list)
            response = await self._fetch_adapting_params(self._fetch_response, message_list, kwargs)
            txt_response = self._extract_content(response)
            cost = self._get_response_cost(input_prompt, txt_response, response)
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
        """Get a JSON response from OpenAI."""
        try:
            if not self.response_format_class:
                raise ValueError("Response format is not set")
            message_list = self._prepare_message_list(input_prompt, image_path_list, message_list)
            response = await self._fetch_adapting_params(
                self._fetch_json_response, message_list, kwargs, is_truncated=self._is_truncated
            )
            txt_response = self._extract_content(response)
            cost = self._get_response_cost(input_prompt, txt_response, response)
            return txt_response, cost
        except Exception as e:
            raise ResponseError(f"Error getting JSON response: {str(e)}")

    def _prepare_message_list(
        self,
        input_prompt: str,
        image_path_list: List[str] = [],
        message_list: Optional[List[Dict[str, str]]] = None,
        system_message: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Prepare the message list for the API call."""
        try:
            if message_list:
                if len(image_path_list) == 0:
                    message_list.append({"role": "user", "content": input_prompt})
                else:
                    content = [{"type": "text", "text": input_prompt}]
                    for image_input in image_path_list:
                        content.append({"type": "image_url", "image_url": {"url": self._encode_image(image_input)}})
                    message_list.append({"role": "user", "content": content})
            else:
                if len(image_path_list) == 0:
                    message_list = [
                        {"role": "system", "content": system_message or self.system_prompt},
                        {"role": "user", "content": input_prompt},
                    ]
                else:
                    content = [{"type": "text", "text": input_prompt}]
                    for image_input in image_path_list:
                        content.append({"type": "image_url", "image_url": {"url": self._encode_image(image_input)}})
                    message_list = [
                        {"role": "system", "content": system_message or self.system_prompt},
                        {"role": "user", "content": content},
                    ]
            return message_list
        except Exception as e:
            raise ProviderError(f"Error preparing message list: {str(e)}")

    async def _fetch_response(self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """Fetch the raw response from OpenAI."""
        try:
            return await self.client.chat.completions.create(model=self.model, messages=message_list, **(kwargs or {}))
        except Exception as e:
            raise ResponseError(f"Error fetching response: {str(e)}") from e

    async def _fetch_json_response(
        self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fetch the JSON response from OpenAI."""
        try:
            # Newer openai releases expose parse() on chat.completions; older ones only have it under beta.
            completions = self.client.chat.completions
            if not hasattr(completions, "parse"):
                completions = self.client.beta.chat.completions
            return await completions.parse(
                model=self.model, messages=message_list, response_format=self.response_format_class, **(kwargs or {})
            )
        except Exception as e:
            raise ResponseError(f"Error fetching JSON response: {str(e)}") from e

    def _extract_content(self, response: Any) -> str:
        """Extract content from the response."""
        try:
            if not response:
                raise ValueError("Empty response received")
            self.last_response = response
            return response.choices[0].message.content
        except Exception as e:
            raise ResponseError(f"Error extracting content: {str(e)}")

    def _is_truncated(self, response_or_error: Any) -> bool:
        """Check whether a response (or the error raised for it) was cut off by the token limit."""
        if isinstance(response_or_error, Exception):
            return isinstance(response_or_error.__cause__ or response_or_error, openai.LengthFinishReasonError)
        choices = getattr(response_or_error, "choices", None)
        return bool(choices) and choices[0].finish_reason == "length"

    def _get_response_cost(self, input_prompt: str, txt_response: str, response: Any) -> Dict[str, float]:
        """Calculate the cost from the token usage the API reported, including hidden reasoning tokens."""
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        # Gemini's OpenAI-compatible endpoint leaves thinking tokens out of completion_tokens but not total_tokens.
        completion_tokens = total_tokens - prompt_tokens if prompt_tokens is not None and total_tokens else None
        return self._get_cost(
            input_messages=input_prompt,
            completion_text=txt_response or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            custom_llm_provider="gemini" if "gemini" in self.model.lower() else "openai",
        )

    # Function to encode the image
    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:{self._image_mime_type(image_path)};base64,{base64_image}"

    def _check_basemodel_class(self, arg):
        """Check if the argument is a Pydantic BaseModel class."""
        return inspect.isclass(arg) and issubclass(arg, BaseModel)
