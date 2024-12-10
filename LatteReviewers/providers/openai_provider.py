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
        self.system_message = data.get('system_message', "You are a helpful assistant.")
        self.output_json_format = data.get('output_json_format', None)
        self.temperature = data.get('temperature', 0.5)
        self.max_tokens = data.get('max_tokens', 150)

    def create_client(self):
        return openai.AsyncOpenAI(api_key=self.api_key)

    async def get_response(self, message: str, message_list: list = None, kwargs: dict=dict()):
        message_list = self._prepare_message_list(message, message_list)
        response = await self._fetch_response(message_list, kwargs)
        return self._extract_content(response)

    async def get_json_response(self, message: str, message_list: list = None, kwargs: dict=dict()):
        if not self.output_json_format:
            raise ValueError("Output JSON format is not set")
        message_list = self._prepare_message_list(message, message_list)
        response = await self._fetch_json_response(message_list, kwargs)
        return self._extract_content(response)

    def _prepare_message_list(self, message: str, message_list: list = None) -> list:
        if message_list:
            message_list.append({"role": "user", "content": message})
        else:
            message_list = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": message}
            ]
        return message_list

    async def _fetch_response(self, message_list: list, kwargs: dict=dict()):
        try:
            return await self.client.chat.completions.create(
                model=self.model,
                messages=message_list,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
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
                max_tokens=self.max_tokens,
                temperature=self.temperature,
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
