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


class GoogleProvider(BaseProvider):
    provider: str = "Google"
    api_key: str = None
    model: str = "gemini-2.5-pro-preview-05-06"
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

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            # Create a config object for generation parameters
            config = None
            if kwargs:
                config = types.GenerateContentConfig()
                # Copy any valid parameters from kwargs to config
                if "temperature" in kwargs:
                    config.temperature = kwargs["temperature"]
                if "top_p" in kwargs:
                    config.top_p = kwargs["top_p"]
                if "top_k" in kwargs:
                    config.top_k = kwargs["top_k"]
                if "max_output_tokens" in kwargs:
                    config.max_output_tokens = kwargs["max_output_tokens"]
                if "safety_settings" in kwargs:
                    config.safety_settings = kwargs["safety_settings"]

            # Call the generate_content method with the proper parameter structure
            response = await loop.run_in_executor(
                None, lambda: self.client.models.generate_content(model=self.model, contents=contents, config=config)
            )

            # Extract text from response
            txt_response = response.text
            self.last_response = response

            # Calculate costs
            cost = await self._get_cost(input_messages=input_prompt, completion_text=txt_response)
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

            # Create a config object for generation parameters
            config = types.GenerateContentConfig()

            # Set the response schema and mime type for JSON responses
            config.response_mime_type = "application/json"
            # Use the response_format dictionary directly
            config.response_schema = self.response_format

            # Copy any valid parameters from kwargs to config
            if kwargs:
                if "temperature" in kwargs:
                    config.temperature = kwargs["temperature"]
                if "top_p" in kwargs:
                    config.top_p = kwargs["top_p"]
                if "top_k" in kwargs:
                    config.top_k = kwargs["top_k"]
                if "max_output_tokens" in kwargs:
                    config.max_output_tokens = kwargs["max_output_tokens"]
                if "safety_settings" in kwargs:
                    config.safety_settings = kwargs["safety_settings"]

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.client.models.generate_content(model=self.model, contents=contents, config=config)
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
            cost = await self._get_cost(input_messages=input_prompt, completion_text=txt_response)
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
                            inline_data=types.Blob(mime_type=f"image/{image_path.split('.')[-1]}", data=image_bytes)
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
                                inline_data=types.Blob(mime_type=f"image/{image_path.split('.')[-1]}", data=image_bytes)
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

    async def _get_cost(self, input_messages: str, completion_text: str) -> Dict[str, float]:
        """Calculate the cost using Gemini's native token counting."""
        try:
            if not self.calculate_cost:
                return {
                    "input_cost": 0.0,
                    "output_cost": 0.0,
                    "total_cost": 0.0,
                }

            # Get input token count using Gemini's native counter
            input_token_response = await self.client.aio.models.count_tokens(
                model=self.model,
                contents=input_messages,
            )
            input_tokens = input_token_response.total_tokens

            # Get output token count
            output_token_response = await self.client.aio.models.count_tokens(
                model=self.model,
                contents=completion_text,
            )
            output_tokens = output_token_response.total_tokens

            # Calculate costs based on Gemini's pricing model
            # Rates as of May 2025 (these should be updated if Google changes pricing)
            if "1.5-pro" in self.model or "1.5-flash" in self.model:
                # Gemini 1.5 Pro/Flash rates
                input_cost = input_tokens * 0.000007  # $0.007 per 1K input tokens
                output_cost = output_tokens * 0.000021  # $0.021 per 1K output tokens
            elif "1.5-ultra" in self.model:
                # Gemini 1.5 Ultra rates
                input_cost = input_tokens * 0.000014  # $0.014 per 1K input tokens
                output_cost = output_tokens * 0.000042  # $0.042 per 1K output tokens
            elif "2.5-pro" in self.model:
                # Gemini 2.5 Pro rates
                input_cost = input_tokens * 0.000025  # $0.025 per 1K input tokens
                output_cost = output_tokens * 0.000075  # $0.075 per 1K output tokens
            elif "2.5-flash" in self.model:
                # Gemini 2.5 Flash rates
                input_cost = input_tokens * 0.000005  # $0.005 per 1K input tokens
                output_cost = output_tokens * 0.000015  # $0.015 per 1K output tokens
            else:  # Default for other/newer models
                input_cost = input_tokens * 0.00001  # $0.01 per 1K input tokens (default)
                output_cost = output_tokens * 0.00003  # $0.03 per 1K output tokens (default)

            total_cost = input_cost + output_cost

            return {
                "input_cost": float(input_cost),
                "output_cost": float(output_cost),
                "total_cost": float(total_cost),
            }
        except Exception as e:
            # If there's an error, log it but don't fail the entire operation
            print(f"Error calculating costs: {str(e)}")
            return {
                "input_cost": -1.0,
                "output_cost": -1.0,
                "total_cost": -1.0,
            }

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

        # Create a config object for generation parameters
        config = None
        if kwargs:
            config = types.GenerateContentConfig()
            # Copy any valid parameters from kwargs to config
            if "temperature" in kwargs:
                config.temperature = kwargs["temperature"]
            if "top_p" in kwargs:
                config.top_p = kwargs["top_p"]
            if "top_k" in kwargs:
                config.top_k = kwargs["top_k"]
            if "max_output_tokens" in kwargs:
                config.max_output_tokens = kwargs["max_output_tokens"]
            if "safety_settings" in kwargs:
                config.safety_settings = kwargs["safety_settings"]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.client.models.generate_content(model=self.model, contents=contents, config=config)
        )

    async def _fetch_json_response(
        self, message_list: List[Dict[str, Any]], kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fetch JSON response (compatibility method)."""
        # This is just a wrapper around get_json_response for backward compatibility
        contents = self._convert_to_genai_format("", [], message_list)

        # Create a config object for generation parameters
        config = types.GenerateContentConfig()

        # Set the response schema and mime type for JSON responses
        config.response_mime_type = "application/json"
        config.response_schema = self.response_format_class

        # Copy any valid parameters from kwargs to config
        if kwargs:
            if "temperature" in kwargs:
                config.temperature = kwargs["temperature"]
            if "top_p" in kwargs:
                config.top_p = kwargs["top_p"]
            if "top_k" in kwargs:
                config.top_k = kwargs["top_k"]
            if "max_output_tokens" in kwargs:
                config.max_output_tokens = kwargs["max_output_tokens"]
            if "safety_settings" in kwargs:
                config.safety_settings = kwargs["safety_settings"]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.client.models.generate_content(model=self.model, contents=contents, config=config)
        )

    def _extract_content(self, response: Any) -> str:
        """Extract content from the response."""
        if not response:
            raise ValueError("Empty response received")
        return response.text

    def _check_basemodel_class(self, arg):
        """Check if the argument is a Pydantic BaseModel class."""
        return inspect.isclass(arg) and issubclass(arg, BaseModel)
