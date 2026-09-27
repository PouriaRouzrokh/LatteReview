"""Base class for all API providers with consistent error handling and type hints."""

import mimetypes
import re
from typing import Optional, Any, List, Dict, Union, Callable, Awaitable, ClassVar, Set, Tuple
import pydantic
import litellm
from tokencost import calculate_prompt_cost, calculate_completion_cost

# Request parameters that cap the response length. Reasoning models count hidden reasoning tokens against these caps.
TOKEN_LIMIT_PARAMS = ("max_tokens", "max_completion_tokens", "max_output_tokens")

# Phrases APIs use when rejecting a request parameter, e.g. OpenAI's "Unsupported parameter: 'max_tokens'" and
# "'temperature' does not support 0.1 with this model", or Anthropic's "`temperature` is deprecated for this model".
PARAM_REJECTION_PHRASES = (
    "unsupported",
    "not supported",
    "does not support",
    "deprecated",
    "unrecognized",
    "not permitted",
    "not allowed",
    "unknown parameter",
    "only the default",
)


class ProviderError(Exception):
    """Base exception for provider-related errors."""

    pass


class ClientCreationError(ProviderError):
    """Raised when client creation fails."""

    pass


class ResponseError(ProviderError):
    """Raised when getting a response fails."""

    pass


class InvalidResponseFormatError(ProviderError):
    """Raised when response format is invalid."""

    pass


class ClientNotInitializedError(ProviderError):
    """Raised when client is not initialized."""

    pass


class BaseProvider(pydantic.BaseModel):
    provider: str = "DefaultProvider"
    client: Optional[Any] = None
    api_key: Optional[str] = None
    model: str = "default-model"
    system_prompt: str = "You are a helpful assistant."
    response_format: Optional[Any] = None
    last_response: Optional[Any] = None
    calculate_cost: bool = True  # if False, the cost will be 0 for both input and output
    # Request parameters this provider's model rejected: maps each one to its replacement name, or None to drop it.
    param_adjustments: Dict[str, Optional[str]] = {}
    _warned: ClassVar[Set[str]] = set()

    class Config:
        arbitrary_types_allowed = True

    def create_client(self) -> Any:
        """Create and initialize the client for the provider."""
        raise NotImplementedError("Subclasses must implement create_client")

    def set_response_format(self, response_format: Dict[str, Any]) -> None:
        """Set the response format for the provider."""
        raise NotImplementedError("Subclasses must implement set_response_format")

    async def get_response(
        self,
        input_prompt: str,
        image_path_list: List[str],
        message_list: Optional[List[Dict[str, str]]] = None,
        system_message: Optional[str] = None,
    ) -> tuple[Any, Dict[str, float]]:
        """Get a response from the provider."""
        raise NotImplementedError("Subclasses must implement get_response")

    async def get_json_response(
        self,
        input_prompt: str,
        image_path_list: List[str],
        message_list: Optional[List[Dict[str, str]]] = None,
        system_message: Optional[str] = None,
    ) -> tuple[Any, Dict[str, float]]:
        """Get a JSON-formatted response from the provider."""
        raise NotImplementedError("Subclasses must implement get_json_response")

    def _prepare_message_list(
        self,
        input_prompt: str,
        image_path_list: List[str],
        message_list: Optional[List[Dict[str, str]]] = None,
        system_message: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Prepare the list of messages to be sent to the provider."""
        raise NotImplementedError("Subclasses must implement _prepare_message_list")

    async def _fetch_response(self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """Fetch the raw response from the provider."""
        raise NotImplementedError("Subclasses must implement _fetch_response")

    async def _fetch_json_response(
        self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fetch the JSON-formatted response from the provider."""
        raise NotImplementedError("Subclasses must implement _fetch_json_response")

    def _extract_content(self, response: Any) -> Any:
        """Extract content from the provider's response."""
        raise NotImplementedError("Subclasses must implement _extract_content")

    def _get_cost(
        self,
        input_messages: List[str],
        completion_text: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, float]:
        """Calculate the cost of a prompt completion.

        Uses the token counts reported by the API (which include hidden reasoning tokens) when available, priced
        with LiteLLM's model map, and falls back to estimating from the text with tokencost. A model missing from
        both pricing maps (e.g., a newly released one) costs 0 with a warning instead of failing the review.
        """
        input_cost, output_cost = 0.0, 0.0
        if self.calculate_cost:
            try:
                if prompt_tokens is not None and completion_tokens is not None:
                    try:
                        input_cost, output_cost = litellm.cost_per_token(
                            model=self.model,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            custom_llm_provider=custom_llm_provider,
                        )
                    except Exception:
                        input_cost = calculate_prompt_cost(input_messages, self.model)
                        output_cost = calculate_completion_cost(completion_text, self.model)
                else:
                    input_cost = calculate_prompt_cost(input_messages, self.model)
                    output_cost = calculate_completion_cost(completion_text, self.model)
            except Exception as e:
                self._warn_once(
                    f"cost:{self.model}",
                    f"could not calculate cost for model <{self.model}>: {str(e).splitlines()[0]}. "
                    "Reporting cost as 0.",
                )
                input_cost, output_cost = 0.0, 0.0
        return {
            "input_cost": float(input_cost),
            "output_cost": float(output_cost),
            "total_cost": float(input_cost + output_cost),
        }

    def _image_mime_type(self, image_path: str) -> str:
        """Return the image's MIME type, e.g. image/jpeg for .jpg (Anthropic rejects the nonstandard image/jpg)."""
        return mimetypes.guess_type(image_path)[0] or f"image/{image_path.split('.')[-1].lower()}"

    def _warn_once(self, key: str, message: str) -> None:
        """Print a warning the first time it occurs in this session."""
        if key not in BaseProvider._warned:
            BaseProvider._warned.add(key)
            print(f"Warning: {message}")

    def _apply_param_adjustments(
        self, kwargs: Optional[Dict[str, Any]], adjustments: Optional[Dict[str, Optional[str]]] = None
    ) -> Dict[str, Any]:
        """Return a copy of kwargs with the parameters this model rejected dropped or renamed."""
        adjusted = dict(kwargs or {})
        for param, replacement in (self.param_adjustments if adjustments is None else adjustments).items():
            if param in adjusted:
                value = adjusted.pop(param)
                if replacement and replacement not in adjusted:
                    adjusted[replacement] = value
        return adjusted

    def _find_rejected_param(
        self, error: Exception, kwargs: Dict[str, Any], adjustments: Dict[str, Optional[str]]
    ) -> Optional[Tuple[str, Optional[str]]]:
        """If the error says the API rejected one of the kwargs, return (param, replacement name or None)."""
        message = str(error)
        if not any(phrase in message.lower() for phrase in PARAM_REJECTION_PHRASES):
            return None
        for param in kwargs:
            if param == "response_format":  # set by LatteReview itself, never user-supplied
                continue
            if getattr(error, "param", None) == param or re.search(rf"(?<![\w.]){re.escape(param)}(?!\w)", message):
                replacement = re.search(r"use ['\"`]?(\w+)['\"`]? instead", message, re.IGNORECASE)
                replacement = replacement.group(1) if replacement else None
                if replacement in kwargs or replacement in adjustments:
                    replacement = None
                return param, replacement
        return None

    async def _fetch_adapting_params(
        self,
        fetch: Callable[[List[Dict[str, Any]], Dict[str, Any]], Awaitable[Any]],
        message_list: List[Dict[str, Any]],
        kwargs: Optional[Dict[str, Any]] = None,
        is_truncated: Optional[Callable[[Any], bool]] = None,
    ) -> Any:
        """Call fetch(message_list, kwargs), adapting to request parameters the model does not accept.

        Newer models reject some parameters that older ones accept: OpenAI's GPT-5 family and o-series and
        Anthropic's Claude Opus 4.7+ and Claude 5 family reject non-default `temperature`/`top_p`, and OpenAI's
        reasoning models renamed `max_tokens` to `max_completion_tokens`. When the API rejects a parameter, it is
        dropped (or renamed, if the API names a replacement) and the call is retried. Models that accept the
        parameters are called exactly as before.

        If is_truncated(response_or_error) reports that the response hit a token limit (reasoning models spend
        hidden tokens against it, so small caps like 200 leave no room for the answer), the call is retried once
        without the limit.

        Adjustments are remembered for this provider, with a one-time warning, only once an adapted call succeeds,
        so an unrelated failure never changes what later calls send.
        """
        kwargs = kwargs or {}
        adjustments = dict(self.param_adjustments)
        warnings = []
        for _ in range(len(kwargs) + 2):
            call_kwargs = self._apply_param_adjustments(kwargs, adjustments)
            token_limits = [p for p in TOKEN_LIMIT_PARAMS if p in call_kwargs]
            try:
                response = await fetch(message_list, call_kwargs)
            except Exception as e:
                rejected = self._find_rejected_param(e, call_kwargs, adjustments)
                if rejected:
                    param, replacement = rejected
                    adjustments[param] = replacement
                    action = f"sending `{replacement}` instead" if replacement else "dropping it"
                    warnings.append(
                        (f"param:{self.model}:{param}", f"model <{self.model}> does not accept `{param}`; {action}.")
                    )
                    continue
                if token_limits and is_truncated and is_truncated(e):
                    adjustments.update(dict.fromkeys(token_limits))
                    warnings.append(self._truncation_warning(token_limits))
                    continue
                raise
            if token_limits and is_truncated and is_truncated(response):
                adjustments.update(dict.fromkeys(token_limits))
                warnings.append(self._truncation_warning(token_limits))
                continue
            self.param_adjustments.update(adjustments)
            for key, message in warnings:
                self._warn_once(key, message)
            return response
        raise ResponseError(f"Model <{self.model}> kept rejecting request parameters: {list(kwargs)}")

    def _truncation_warning(self, token_limits: List[str]) -> Tuple[str, str]:
        """Build the warning for a response that was cut off by the given token limits."""
        return (
            f"truncated:{self.model}",
            f"model <{self.model}> ran out of tokens before finishing its answer (reasoning models count hidden "
            f"reasoning against the limit); retried without {', '.join(f'`{p}`' for p in token_limits)}.",
        )
