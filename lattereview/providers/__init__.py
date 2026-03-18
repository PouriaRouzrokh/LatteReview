import warnings

warnings.warn(
    "lattereview.providers is deprecated and will be removed in v3.0. "
    "v2 uses Pydantic AI model strings directly (e.g., 'openai:gpt-4o', "
    "'anthropic:claude-sonnet-4-6') — no provider wrappers needed. "
    "See lattereview.agentic for the new API.",
    DeprecationWarning,
    stacklevel=2,
)

from .litellm_provider import LiteLLMProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .google_provider import GoogleProvider
