import os
import openai

from .base_provider import BaseProvider

class OpenAIProvider(BaseProvider):
    def __init__(self, **data):
        super().__init__(**data)
        self.provider = "OpenAI"
        self.api_key = data.get('api_key', os.getenv("OPENAI_API_KEY"))
        self.client = self.create_client()
        self.model = data.get('model', "gpt-4o-mini")
        self.output_json_format = data.get('output_json_format', None)

    def create_client(self):
        return openai.AsyncOpenAI(api_key=self.api_key)

    async def get_response(self, message: str, message_list: list = None, system_message: str = None, **kwargs):
        message_list = self._prepare_message_list(message, message_list, system_message)
        response = await self._fetch_response(message_list, kwargs)
        return self._extract_content(response)

    async def get_json_response(self, message: str, message_list: list = None, system_message: str = None, **kwargs):
        if not self.output_json_format:
            raise ValueError("Output JSON format is not set")
        message_list = self._prepare_message_list(message, message_list, system_message)
        response = await self._fetch_json_response(message_list, kwargs)
        return self._extract_content(response)

    def _prepare_message_list(self, message: str, message_list: list = None, system_message: str = None) -> list:
        if message_list:
            message_list.append({"role": "user", "content": message})
            if system_message:
                message_list.insert(0, {"role": "system", "content": system_message})
        else:
            if not system_message:
                system_message = "You are a helpful assistant."
            message_list = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message}
            ]
        return message_list

    async def _fetch_response(self, message_list: list, kwargs: dict=dict()):
        try:
            return await self.client.chat.completions.create(
                model=self.model,
                messages=message_list,
                **kwargs
            )
        except Exception as e:
            print(f"Error: {e}")
            return None

    async def _fetch_json_response(self, message_list: list, kwargs: dict=dict()):
        try:
            return await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=message_list,
                response_format=self.output_json_format,
                **kwargs
            )
        except Exception as e:
            print(f"Error: {e}")
            return None

    def _extract_content(self, response):
        if response:
            self.last_response = response
            return response.choices[0].message.content
        return None
