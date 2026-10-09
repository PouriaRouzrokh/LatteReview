"""LiteLLM API provider implementation with comprehensive error handling and type safety."""

import base64
import inspect
from typing import Optional, List, Dict, Any, Union, Tuple, Type
import json
import re
from pydantic import BaseModel, create_model
import litellm
from litellm import acompletion, completion_cost
from .base_provider import BaseProvider, ProviderError, ResponseError, InvalidResponseFormatError

litellm.drop_params = True  # Drop unsupported parameters from the API
litellm.enable_json_schema_validation = True  # Enable client-side JSON schema validation

PERPLEXITY_PREFIX = "perplexity/"
PERPLEXITY_NUM_RETRIES = 5  # LiteLLM's retries back off between attempts, unlike the reviewer's own retries


class LiteLLMProvider(BaseProvider):
    provider: str = "LiteLLM"
    model: str = "gpt-6-luna"
    custom_llm_provider: Optional[str] = None
    response_format_class: Optional[Any] = None

    def __init__(self, custom_llm_provider: Optional[str] = None, **data: Any) -> None:
        """Initialize the LiteLLM provider."""
        data_with_provider = {**data}
        if custom_llm_provider:
            data_with_provider["custom_llm_provider"] = custom_llm_provider

        super().__init__(**data_with_provider)

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

    def _safe_completion_cost(self, response: Any) -> float:
        """Calculate the completion cost, returning 0.0 when the model is missing from
        LiteLLM's pricing map (e.g., proxied or newly released models) so that a
        successful review is never discarded over cost bookkeeping.

        A cost reported by the API itself wins over token pricing, because it includes per-request
        fees that token prices miss (e.g., the web-search fee of Perplexity's Sonar models)."""
        reported = self._reported_cost(response)
        if reported is not None:
            return reported
        try:
            try:
                return completion_cost(completion_response=response)
            except Exception:
                # Some responses (e.g., Groq's) report a model name that is not the key in LiteLLM's pricing map,
                # so price the reported token usage under the model name the user gave instead.
                input_cost, output_cost = litellm.cost_per_token(
                    model=self.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    custom_llm_provider=self.custom_llm_provider,
                )
                return input_cost + output_cost
        except Exception as e:
            self._warn_once(
                f"cost:{self.model}",
                f"could not calculate cost for model <{self.model}>: {str(e).splitlines()[0]}. Reporting cost as 0.",
            )
            return 0.0

    def _route(self, kwargs: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """Return the LiteLLM model name and call arguments for this provider's model.

        Perplexity no longer serves chat completions: its models (e.g., "perplexity/sonar") and the other vendors' models
        it hosts (e.g., "perplexity/openai/gpt-6-luna") are reached through its Agent API, via LiteLLM's Responses bridge.
        New Perplexity accounts allow about one request per second, so rate-limited calls are retried with backoff.
        """
        kwargs = dict(kwargs or {})
        prefix = PERPLEXITY_PREFIX
        if not self.model.startswith(prefix) or self.model.startswith(prefix + "responses/"):
            return self.model, kwargs
        name = self.model[len(prefix) :]
        kwargs.setdefault("num_retries", PERPLEXITY_NUM_RETRIES)
        return f"{prefix}responses/{name if '/' in name else prefix + name}", kwargs

    @staticmethod
    def _reported_cost(response: Any) -> Optional[float]:
        """Return the cost the API reported in `usage.cost` (OpenRouter: a number; Perplexity: {"total_cost": ...}),
        or None if it reported none. A zero cost (e.g., OpenRouter with your own provider key) counts as none."""
        cost = getattr(getattr(response, "usage", None), "cost", None)
        if isinstance(cost, dict):
            cost = cost.get("total_cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost > 0:
            return float(cost)
        return None

    async def get_response(
        self,
        input_prompt: str,
        image_path_list: List[str] = [],
        message_list: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, float]]:
        """Get a response from LiteLLM."""
        try:
            message_list = self._prepare_message_list(input_prompt, image_path_list, message_list)
            response = await self._fetch_adapting_params(self._fetch_response, message_list, kwargs)
            txt_response = self._extract_content(response)
            cost = self._safe_completion_cost(response)

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
        """Get a JSON response from LiteLLM using the defined schema."""
        try:
            if not self.response_format_class:
                raise ValueError("Response format is not set")

            message_list = self._prepare_message_list(input_prompt, image_path_list, message_list)
            response = await self._fetch_adapting_params(
                self._fetch_json_response, message_list, kwargs, is_truncated=self._is_truncated
            )
            txt_response = self._extract_content(response)

            # Parse the response as JSON if it's a string. In basic JSON mode some models
            # (e.g., Claude) wrap the JSON in a markdown code fence.
            if isinstance(txt_response, str):
                fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", txt_response, re.DOTALL)
                txt_response = json.loads(fenced.group(1) if fenced else txt_response)

            cost = self._safe_completion_cost(response)

            return txt_response, cost
        except Exception as e:
            raise ResponseError(f"Error getting JSON response: {str(e)}")

    async def _fetch_json_response(
        self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fetch a response in the defined JSON schema, falling back to basic JSON mode if the provider rejects it."""
        # Pass response format directly to acompletion
        kwargs = {**(kwargs or {}), "response_format": self.response_format_class}
        try:
            return await self._fetch_response(message_list, kwargs)
        except Exception as e:
            # Some providers (e.g., DeepSeek) no longer accept json_schema response
            # formats, and some models (e.g., Claude Opus 5.5 and Fable 5.1) reject the
            # forced tool call that older LiteLLM releases use to emulate them. Retry in
            # basic JSON mode with an explicit JSON instruction (providers like DeepSeek
            # require the word "json" in the prompt).
            if "response_format" not in str(e) and "tool_choice" not in str(e):
                raise
            fallback_kwargs = {**kwargs, "response_format": {"type": "json_object"}}
            json_keys = ", ".join(self.response_format_class.model_fields.keys())
            fallback_messages = message_list + [
                {
                    "role": "user",
                    "content": f"Return your response as a valid JSON object with these keys: {json_keys}.",
                }
            ]
            return await self._fetch_response(fallback_messages, fallback_kwargs)

    def _is_truncated(self, response_or_error: Any) -> bool:
        """Check whether a response (or the error raised for it) was cut off by the token limit."""
        if isinstance(response_or_error, Exception):
            # LiteLLM validates the JSON before returning it, so a cut-off answer surfaces as a validation error;
            # some providers (e.g., Groq) reject cut-off JSON themselves.
            error = response_or_error.__cause__ or response_or_error
            return isinstance(error, litellm.JSONSchemaValidationError) or "max completion tokens reached" in str(error)
        choices = getattr(response_or_error, "choices", None)
        return bool(choices) and choices[0].finish_reason == "length"

    def _prepare_message_list(
        self,
        input_prompt: str,
        image_path_list: List[str],
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
        """Fetch the raw response from LiteLLM."""
        try:
            model, kwargs = self._route(kwargs)
            response = await acompletion(
                model=model, messages=message_list, custom_llm_provider=self.custom_llm_provider, **kwargs
            )
            return response
        except Exception as e:
            raise ResponseError(f"Error fetching response: {str(e)}") from e

    def _extract_content(self, response: Any) -> str:
        """Extract content from the response, handling both direct content and tool calls."""
        try:
            if not response:
                raise ValueError("Empty response received")

            self.last_response = response
            response_message = response.choices[0].message

            # Check for direct content first
            if response_message.content is not None:
                return response_message.content

            # Check for tool calls if content is None
            if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "json_tool_call":
                        return tool_call.function.arguments

            raise ValueError("No content or valid tool calls found in response")

        except Exception as e:
            raise ResponseError(f"Error extracting content: {str(e)}")

    # Function to encode the image
    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:{self._image_mime_type(image_path)};base64,{base64_image}"

    def _check_basemodel_class(self, arg):
        """Check if the argument is a Pydantic BaseModel class."""
        return inspect.isclass(arg) and issubclass(arg, BaseModel)
