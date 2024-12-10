import os
from typing import Optional, List, Dict, Any
from .base_provider import BaseProvider
import openai


class OpenAIProvider(BaseProvider):
    def __init__(self, **data):
        """
        Initialize the OpenAIProvider.
        """
        super().__init__(**data)
        self.provider = "OpenAI"
        self.api_key = data.get("api_key", os.getenv("OPENAI_API_KEY"))
        self.client = self.create_client()
        self.model = data.get("model", "gpt-4o-mini")
        self.output_json_format = data.get("output_json_format", None)

    def create_client(self) -> openai.AsyncOpenAI:
        """
        Create and return the OpenAI client.
        """
        return openai.AsyncOpenAI(api_key=self.api_key)

    async def get_response(
        self,
        messages: str,
        message_list: Optional[List[Dict[str, str]]] = None,
        system_message: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Get a response from OpenAI.
        """
        message_list = self._prepare_message_list(messages, message_list, system_message)
        response = await self._fetch_response(message_list, kwargs)
        return self._extract_content(response)

    async def get_json_response(
        self,
        messages: str,
        message_list: Optional[List[Dict[str, str]]] = None,
        system_message: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Get a JSON response from OpenAI.
        """
        if not self.output_json_format:
            raise ValueError("Output JSON format is not set")
        message_list = self._prepare_message_list(messages, message_list, system_message)
        response = await self._fetch_json_response(message_list, kwargs)
        return self._extract_content(response)

    def _prepare_message_list(
        self,
        message: str,
        message_list: Optional[List[Dict[str, str]]] = None,
        system_message: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Prepare the list of messages to send.
        """
        if message_list:
            message_list.append({"role": "user", "content": message})
            if system_message:
                message_list.insert(0, {"role": "system", "content": system_message})
        else:
            if not system_message:
                system_message = "You are a helpful assistant."
            message_list = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ]
        return message_list

    async def _fetch_response(
        self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Fetch the raw response from OpenAI.
        """
        try:
            return await self.client.chat.completions.create(
                model=self.model, messages=message_list, **(kwargs or {})
            )
        except Exception as e:
            print(f"Error fetching response: {e}")
            return None

    async def _fetch_json_response(
        self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Fetch the JSON response from OpenAI.
        """
        try:
            return await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=message_list,
                response_format=self.output_json_format,
                **(kwargs or {}),
            )
        except Exception as e:
            print(f"Error fetching JSON response: {e}")
            return None

    def _extract_content(self, response: Any) -> Any:
        """
        Extract content from the response.
        """
        if response:
            self.last_response = response
            return response.choices[0].message.content
        return None