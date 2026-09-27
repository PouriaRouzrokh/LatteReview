"""Google Gemini API provider implementation with comprehensive error handling and type safety."""

import base64
import inspect
import os
from typing import Optional, List, Dict, Any, Tuple
import asyncio

from pydantic import BaseModel
from google import genai
from google.genai import types

from .base_provider import BaseProvider, ProviderError, ClientCreationError, ResponseError, InvalidResponseFormatError

# model_args keys that are copied into the Gemini GenerateContentConfig; other keys are ignored.
GENERATION_CONFIG_PARAMS = ("temperature", "top_p", "top_k", "max_output_tokens", "safety_settings", "thinking_config")


class GoogleProvider(BaseProvider):
    provider: str = "Google"
    api_key: str = None
    model: str = "gemini-3.8-flash"
    response_format_class: Optional[Any] = None

    def __init__(self, **data: Any) -> None:
        """Initialize the Google provider with error handling."""
        super().__init__(**data)
        try:
            self.client = self.create_client()
        except Exception as e:
            raise ClientCreationError(f"Failed to create Google client: {str(e)}")

    def set_response_format(self, response_format: Any) -> None:
        """Set the response format for JSON responses."""
        try:
            if not response_format:
                raise InvalidResponseFormatError("Response format cannot be empty")

            if isinstance(response_format, dict):
                # Convert dictionary format to proper Google Gemini schema format
                self.response_format = self._dict_to_gemini_schema(response_format)
                self.response_format_class = None
            elif self._check_basemodel_class(response_format):
                # If it's a Pydantic model, extract its schema
                self.response_format_class = response_format
                # Convert Pydantic model to Google Gemini schema format
                self.response_format = self._model_to_gemini_schema(response_format)
        except Exception as e:
            raise ProviderError(f"Error setting response format: {str(e)}")

    def _dict_to_gemini_schema(self, format_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a dictionary format specification to Google Gemini schema format."""
        schema = {"type": "OBJECT", "properties": {}}

        # Process each field in the dictionary
        for field_name, field_type in format_dict.items():
            # Handle list types
            if isinstance(field_type, list) and len(field_type) > 0:
                item_type = field_type[0]

                # If the item is a dictionary, it's a nested object
                if isinstance(item_type, dict):
                    schema["properties"][field_name] = {
                        "type": "ARRAY",
                        "items": self._dict_to_gemini_schema(item_type),
                    }
                else:
                    # For primitive types in arrays
                    schema["properties"][field_name] = {
                        "type": "ARRAY",
                        "items": {"type": self._get_gemini_type(item_type)},
                    }
            # Handle nested dictionaries (objects)
            elif isinstance(field_type, dict):
                schema["properties"][field_name] = self._dict_to_gemini_schema(field_type)
            # Handle primitive types
            else:
                schema["properties"][field_name] = {"type": self._get_gemini_type(field_type)}

        # All properties are required by default
        schema["required"] = list(format_dict.keys())

        return schema

    def _model_to_gemini_schema(self, model_class) -> Dict[str, Any]:
        """Convert a Pydantic model to Google Gemini schema format."""
        # Extract field information from the model
        fields = {}
        for field_name, field in model_class.__annotations__.items():
            if hasattr(field, "__origin__") and field.__origin__ is list:
                # Handle List type
                if hasattr(field, "__args__") and len(field.__args__) > 0:
                    item_type = field.__args__[0]
                    fields[field_name] = [item_type]
            else:
                fields[field_name] = field

        # Convert the extracted fields to Gemini schema format
        return self._dict_to_gemini_schema(fields)

    def _get_gemini_type(self, python_type) -> str:
        """Map Python types to Google Gemini schema types."""
        type_mapping = {str: "STRING", int: "INTEGER", float: "NUMBER", bool: "BOOLEAN"}

        # If it's a type object (like str, int), look it up in the mapping
        if python_type in type_mapping:
            return type_mapping[python_type]

        # For class objects, check if they match any of our mapped types
        for py_type, gemini_type in type_mapping.items():
            if python_type == py_type:
                return gemini_type

        # Default to STRING for unknown types
        return "STRING"

    def create_client(self) -> genai.Client:
        """Create and return the Google genai client."""
        try:
            if not self.api_key:
                self.api_key = os.getenv("GEMINI_API_KEY")
                if not self.api_key:
                    raise ClientCreationError(
                        "GEMINI_API_KEY environment variable is not set. Please pass your API key or set this variable."
                    )

            return genai.Client(api_key=self.api_key)
        except Exception as e:
            raise ClientCreationError(f"Failed to create Google genai client: {str(e)}")

    async def get_response(
        self,
        input_prompt: str,
        image_path_list: List[str] = [],
        message_list: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, float]]:
        """Get a response from Google Gemini."""
        try:
            # Convert to Google genai format
            contents = self._convert_to_genai_format(input_prompt, image_path_list, message_list)

            response = await self._fetch_adapting_params(self._generate, contents, self._config_kwargs(kwargs))

            # Extract text from response
            txt_response = response.text
            self.last_response = response

            # Calculate costs
            cost = self._get_response_cost(input_prompt, txt_response, response)
            return txt_response, cost

        except Exception as e:
            # Add more detailed error information for debugging
            error_msg = f"Error getting response: {str(e)}"
            import traceback

            error_msg += f"\nTraceback: {traceback.format_exc()}"
            raise ResponseError(error_msg)

    async def get_json_response(
        self,
        input_prompt: str,
        image_path_list: List[str] = [],
        message_list: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, float]]:
        """Get a JSON response from Google Gemini."""
        try:
            if not self.response_format:
                raise ValueError("Response format is not set")

            # Convert to Google genai format
            contents = self._convert_to_genai_format(input_prompt, image_path_list, message_list)

            response = await self._fetch_adapting_params(
                self._generate_json, contents, self._config_kwargs(kwargs), is_truncated=self._is_truncated
            )

            self.last_response = response

            # Extract structured data
            if hasattr(response, "parsed"):
                parsed_response = response.parsed
                # If the parsed response is a Pydantic model, convert it to a dictionary
                if hasattr(parsed_response, "dict") and callable(getattr(parsed_response, "dict")):
                    parsed_response = parsed_response.dict()
                # If the parsed response is a list of Pydantic models, convert each to a dictionary
                elif isinstance(parsed_response, list) and all(
                    hasattr(item, "dict") and callable(getattr(item, "dict")) for item in parsed_response
                ):
                    parsed_response = [item.dict() for item in parsed_response]
            else:
                # If no parsed attribute, use the response itself
                parsed_response = response

            # Get the text representation for cost calculation
            txt_response = response.text if hasattr(response, "text") else str(parsed_response)

            # Calculate costs
            cost = self._get_response_cost(input_prompt, txt_response, response)
            return parsed_response, cost

        except Exception as e:
            # Add more detailed error information for debugging
            error_msg = f"Error getting JSON response: {str(e)}"
            import traceback

            error_msg += f"\nTraceback: {traceback.format_exc()}"
            raise ResponseError(error_msg)

    def _convert_to_genai_format(
        self,
        input_prompt: str,
        image_path_list: List[str] = [],
        message_list: Optional[List[Dict[str, str]]] = None,
    ) -> Any:
        """Convert input to Google genai format."""
        try:
            # SIMPLE CASE: If just a single text prompt with no images, return as is
            if not message_list and not image_path_list:
                return input_prompt

            # MULTIMODAL CASE: If we have images but no message list, create parts list with text and images
            if not message_list and image_path_list:
                parts = []
                # Add text part
                parts.append(types.Part(text=input_prompt))
                # Add image parts
                for image_path in image_path_list:
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                    parts.append(
                        types.Part(
                            inline_data=types.Blob(mime_type=self._image_mime_type(image_path), data=image_bytes)
                        )
                    )
                return parts

            # CHAT CASE: If we have a message list, convert to Content objects
            if message_list:
                # Start with any existing messages
                contents = []
                system_message = None

                # Process each message
                for message in message_list:
                    role = message.get("role", "user")
                    content = message.get("content", "")

                    # Capture system message for special handling
                    if role == "system":
                        system_message = content
                        continue

                    # Map roles: OpenAI -> Gemini
                    role = "model" if role == "assistant" else "user"

                    # Handle text content
                    if isinstance(content, str):
                        contents.append(types.Content(role=role, parts=[types.Part(text=content)]))
                    # Handle multimodal content
                    elif isinstance(content, list):
                        parts = []
                        for part in content:
                            if part.get("type") == "text":
                                parts.append(types.Part(text=part.get("text", "")))
                            elif part.get("type") == "image":
                                image_data = None
                                if "image" in part and "data" in part["image"]:
                                    image_data = part["image"]["data"]
                                parts.append(
                                    types.Part(
                                        inline_data=types.Blob(mime_type="image/jpeg", data=image_data)  # Assume JPEG
                                    )
                                )

                        if parts:
                            contents.append(types.Content(role=role, parts=parts))

                # Add the current prompt as the final user message
                if image_path_list:
                    # Add multi-modal parts
                    parts = [types.Part(text=input_prompt)]
                    for image_path in image_path_list:
                        with open(image_path, "rb") as f:
                            image_bytes = f.read()
                        parts.append(
                            types.Part(
                                inline_data=types.Blob(mime_type=self._image_mime_type(image_path), data=image_bytes)
                            )
                        )
                    contents.append(types.Content(role="user", parts=parts))
                else:
                    # Add text-only message
                    contents.append(types.Content(role="user", parts=[types.Part(text=input_prompt)]))

                # Add system prompt to the beginning if provided
                if system_message:
                    # Prepend to the first user message since Gemini doesn't support system messages directly
                    for i, content in enumerate(contents):
                        if content.role == "user":
                            prefix = f"System instructions: {system_message}\n\nUser: "
                            if content.parts and hasattr(content.parts[0], "text"):
                                content.parts[0].text = prefix + content.parts[0].text
                            break
                    else:
                        # If no user message found, add a new one with system instructions
                        contents.insert(
                            0,
                            types.Content(
                                role="user", parts=[types.Part(text=f"System instructions: {system_message}")]
                            ),
                        )

                return contents

            return input_prompt  # Fallback to simple text if all else fails

        except Exception as e:
            raise ProviderError(f"Error converting to genai format: {str(e)}")

    def _config_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only the model_args that map onto the Gemini generation config."""
        return {key: value for key, value in kwargs.items() if key in GENERATION_CONFIG_PARAMS}

    def _build_config(self, kwargs: Optional[Dict[str, Any]] = None, json_mode: bool = False) -> Any:
        """Build the generation config from model_args, plus the response schema for JSON responses."""
        if not kwargs and not json_mode:
            return None
        config = types.GenerateContentConfig(**self._config_kwargs(kwargs or {}))
        if json_mode:
            config.response_mime_type = "application/json"
            config.response_schema = self.response_format
        return config

    async def _generate(self, contents: Any, kwargs: Optional[Dict[str, Any]] = None, json_mode: bool = False) -> Any:
        """Call generate_content in an executor so the synchronous client does not block the event loop."""
        config = self._build_config(kwargs, json_mode)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.client.models.generate_content(model=self.model, contents=contents, config=config)
        )

    async def _generate_json(self, contents: Any, kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """Call generate_content with the JSON response schema."""
        return await self._generate(contents, kwargs, json_mode=True)

    def _is_truncated(self, response_or_error: Any) -> bool:
        """Check whether a response was cut off by the token limit."""
        candidates = getattr(response_or_error, "candidates", None)
        return bool(candidates) and candidates[0].finish_reason == types.FinishReason.MAX_TOKENS

    def _get_response_cost(self, input_prompt: str, txt_response: str, response: Any) -> Dict[str, float]:
        """Calculate the cost from the token usage Gemini reported; thinking tokens are billed as output."""
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        completion_tokens = None
        if usage is not None and getattr(usage, "candidates_token_count", None) is not None:
            completion_tokens = usage.candidates_token_count + (getattr(usage, "thoughts_token_count", None) or 0)
        return self._get_cost(
            input_messages=input_prompt,
            completion_text=txt_response or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            custom_llm_provider="gemini",
        )

    def _prepare_message_list(
        self,
        input_prompt: str,
        image_path_list: List[str] = [],
        message_list: Optional[List[Dict[str, str]]] = None,
        system_message: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Prepare the message list (compatibility method for base provider)."""
        # Create a new message list if none provided
        if not message_list:
            message_list = []

            # Add system message if provided
            if system_message or self.system_prompt:
                message_list.append({"role": "system", "content": system_message or self.system_prompt})

        # Add the current prompt as a user message
        if not image_path_list:
            message_list.append({"role": "user", "content": input_prompt})
        else:
            content = [{"type": "text", "text": input_prompt}]
            for image_path in image_path_list:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                content.append({"type": "image", "image": {"data": image_bytes}})
            message_list.append({"role": "user", "content": content})

        return message_list

    async def _fetch_response(self, message_list: List[Dict[str, Any]], kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """Fetch response (compatibility method)."""
        # This is just a wrapper around get_response for backward compatibility
        contents = self._convert_to_genai_format("", [], message_list)
        return await self._generate(contents, kwargs)

    async def _fetch_json_response(
        self, message_list: List[Dict[str, Any]], kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fetch JSON response (compatibility method)."""
        # This is just a wrapper around get_json_response for backward compatibility
        contents = self._convert_to_genai_format("", [], message_list)
        return await self._generate_json(contents, kwargs)

    def _extract_content(self, response: Any) -> str:
        """Extract content from the response."""
        if not response:
            raise ValueError("Empty response received")
        return response.text

    def _check_basemodel_class(self, arg):
        """Check if the argument is a Pydantic BaseModel class."""
        return inspect.isclass(arg) and issubclass(arg, BaseModel)
