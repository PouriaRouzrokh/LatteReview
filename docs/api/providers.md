# Providers Module Documentation

This module provides different language model provider implementations for the LatteReview package. It handles interactions with various LLM APIs in a consistent, type-safe manner and now supports image-based inputs in addition to text inputs.

## Overview

The providers module includes:

- `BaseProvider`: Abstract base class defining the provider interface
- `OpenAIProvider`: Implementation for OpenAI API (including GPT models)
- `OllamaProvider`: Implementation for local Ollama models
- `LiteLLMProvider`: Implementation using LiteLLM for unified API access
- `GoogleProvider`: Implementation for Google's Gemini API (including Gemini Pro and Flash models)
- `SystemOneProvider` (new in v1.3.0): Client for System One **decision models** such as TypeSafe's Jev, which answer typed questions with probabilities instead of generating text. It is used by the decision reviewers; see [Decision Models](../decision_models.md).

** You can use any of the models offered by the providers above as far as they support structured outputs. **

## Model compatibility

Newer models, especially reasoning models, reject some request parameters that older models accept:

| Models | Parameters they reject |
| --- | --- |
| OpenAI GPT-5 and GPT-6 families and o-series (e.g., `gpt-6-sol`, `gpt-5.5`, `o4-mini`) | `max_tokens` (renamed to `max_completion_tokens`); most also reject `temperature` other than 1 and `top_p` |
| Anthropic Claude Opus 4.7+, Claude 5 and Fable (e.g., `claude-opus-5-5`, `claude-sonnet-5`, `claude-fable-5-1`) | `temperature`, `top_p`, `top_k` |
| Google Gemini 3.x (e.g., `gemini-3.8-flash`) | None yet, but Google has deprecated `temperature`, `top_p` and `top_k` and recommends leaving them at their defaults |

Since v1.2.0, LatteReview handles these differences for you:

- **Rejected parameters are dropped.** If the API rejects a parameter in `model_args`, the provider drops it (or sends `max_completion_tokens` instead of `max_tokens`), prints a one-time warning, and retries. The adjustment applies to the rest of that provider's calls. Models that accept the parameter are called exactly as before, so code such as `model_args={"max_tokens": 200, "temperature": 0.1}` keeps working with new models; the rejected values are simply not applied.
- **Token limits don't cut off answers.** Reasoning models count hidden reasoning tokens against `max_tokens`/`max_completion_tokens`/`max_output_tokens`, so a small limit can leave no room for the answer. If a structured answer is cut off by the limit, the call is retried once without it (with a warning), and the limit is not sent again for that provider.
- **Claude returns structured output natively.** Through `LiteLLMProvider`, Claude models use Anthropic's native structured outputs. If a model rejects the forced tool call that older LiteLLM releases use instead (Claude Opus 5.5 and Fable 5.1 do), LatteReview falls back to JSON mode.

Recommendations:

- For reasoning models, leave out `max_tokens` and `temperature`. Use `reasoning_effort` to trade depth for speed and cost (e.g., `"low"` for screening, `"high"` for an expert reviewer). It works with `LiteLLMProvider` for OpenAI, Claude and Gemini models, and with `OpenAIProvider` for OpenAI models and Gemini (through Google's OpenAI-compatible endpoint). Models without reasoning (e.g., `gpt-4o-mini`) reject it, and it is dropped with a warning.
- With `GoogleProvider`, use Gemini's own `thinking_config` instead: `{"thinking_level": "low"}` on Gemini 3.x or `{"thinking_budget": 0}` on Gemini 2.5 Flash. `GoogleProvider` passes only `temperature`, `top_p`, `top_k`, `max_output_tokens`, `safety_settings` and `thinking_config` to Gemini and ignores other keys (such as `max_tokens` and `reasoning_effort`).
- Older models (e.g., `gpt-4o-mini`, `gemini-2.5-flash`, `claude-haiku-4-5`) keep accepting `temperature` and `max_tokens`.

Models tested with real API calls for v1.2.0:

| Provider path | Models |
| --- | --- |
| `OpenAIProvider` | `gpt-6-astra`, `gpt-6-sol`, `gpt-6-luna`, `gpt-5.5`, `gpt-4o-mini` |
| `OpenAIProvider` (Gemini endpoint) | `gemini-3.8-flash` |
| `GoogleProvider` | `gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-pro-preview`, `gemini-2.5-pro`, `gemini-2.5-flash` |
| `OllamaProvider` (local, Apple M5, 32 GB) | `qwen3.8:27b` |
| `LiteLLMProvider` | `gpt-6-astra`, `gpt-6-luna`, `gpt-4o-mini`, `o4-mini`, `anthropic/claude-opus-5-5`, `anthropic/claude-opus-5`, `anthropic/claude-sonnet-5`, `anthropic/claude-haiku-4-5`, `anthropic/claude-fable-5-1`, `gemini/gemini-3.8-flash`, `groq/openai/gpt-oss-120b`, `deepseek/deepseek-flash`, `openrouter/qwen/qwen3.8-27b`, `ollama_chat/qwen3.8:27b` |

## BaseProvider

### Description

The `BaseProvider` class serves as the foundation for all LLM provider implementations. It provides a consistent interface and error handling for interacting with language models.

### Class Definition

```python
class BaseProvider(pydantic.BaseModel):
    provider: str = "DefaultProvider"
    client: Optional[Any] = None
    api_key: Optional[str] = None
    model: str = "default-model"
    system_prompt: str = "You are a helpful assistant."
    response_format: Optional[Any] = None
    last_response: Optional[Any] = None
    calculate_cost: bool = True  # Controls whether to calculate token costs
    param_adjustments: Dict[str, Optional[str]] = {}  # Parameters the model rejected (see Model compatibility)
```

### Error Types

```python
class ProviderError(Exception): pass
class ClientCreationError(ProviderError): pass
class ResponseError(ProviderError): pass
class InvalidResponseFormatError(ProviderError): pass
class ClientNotInitializedError(ProviderError): pass
```

### Core Methods

#### `create_client()`

Abstract method for initializing the provider's client.

```python
def create_client(self) -> Any:
    """Create and initialize the client for the provider."""
    raise NotImplementedError
```

#### `get_response()`

Get a text or image response from the model.

```python
async def get_response(
    self,
    input_prompt: str,
    image_path_list: List[str],
    message_list: Optional[List[Dict[str, str]]] = None,
    system_message: Optional[str] = None,
) -> tuple[Any, Dict[str, float]]:
    """Get a response from the provider."""
    raise NotImplementedError
```

#### `get_json_response()`

Get a JSON-formatted response from the model.

```python
async def get_json_response(
    self,
    input_prompt: str,
    image_path_list: List[str],
    message_list: Optional[List[Dict[str, str]]] = None,
    system_message: Optional[str] = None,
) -> tuple[Any, Dict[str, float]]:
    """Get a JSON-formatted response from the provider."""
    raise NotImplementedError
```

### Cost Calculation

All providers include built-in cost calculation for both input and output tokens. Costs are computed from the token usage the API reports, which includes hidden reasoning and thinking tokens, priced with LiteLLM's model price map. When the usage or the price is not available, the cost is estimated from the text with the `tokencost` package. A model missing from both price maps (e.g., one released after your LiteLLM version) is reported as costing 0, with a warning, instead of failing the review. This can be controlled using the `calculate_cost` parameter:

```python
# Initialize provider with cost calculation disabled
provider = BaseProvider(calculate_cost=False)  # All costs will be 0

# Initialize provider with cost calculation enabled (default)
provider = BaseProvider(calculate_cost=True)   # Costs will be calculated based on token usage
```

The cost calculation returns a dictionary with three values:

- `input_cost`: Cost of the input tokens
- `output_cost`: Cost of the output tokens
- `total_cost`: Sum of input and output costs

When `calculate_cost` is False, all costs will be 0.

## OpenAIProvider

### Description

Implementation for OpenAI's API, supporting OpenAI models, Gemini models through a compatible endpoint, and OpenRouter's model marketplace. The updated version supports processing text and image inputs.

### Class Definition

```python
class OpenAIProvider(BaseProvider):
    provider: str = "OpenAI"
    api_key: str = None
    base_url: str = None
    model: str = "gpt-6-luna"
    response_format_class: Optional[Any] = None
```

### Key Features

- Automatic API key handling from environment variables
- Support for OpenAI and Gemini models
- Custom base URL support for alternative endpoints
- OpenRouter integration for access to multiple model providers
- JSON response validation
- Image input handling
- Comprehensive error handling

### Usage Example

```python
from lattereview.providers import OpenAIProvider

# Initialize with OpenAI model
provider = OpenAIProvider(model="gpt-6-sol")

# Initialize with Gemini model (no "gemini/" prefix here; that prefix is only for LiteLLMProvider)
provider = OpenAIProvider(model="gemini-3.8-flash")

# Initialize with OpenRouter
provider = OpenAIProvider(
    model="anthropic/claude-sonnet-4-6",
    api_key="your_openrouter_key",
    base_url="https://openrouter.ai/api/v1"
)

# Get a response
response, cost = await provider.get_response("What is the capital of the country shown in this map?", ["path/to/image.jpg"])

# Get JSON response
provider.set_response_format({"key": str, "value": int})
response, cost = await provider.get_json_response("Format this as JSON", [])
```

### Using OpenRouter

OpenRouter provides access to a wide variety of language models through a unified API endpoint. To use OpenRouter with the OpenAIProvider:

1. Sign up at [OpenRouter](https://openrouter.ai/) to get an API key
2. Set the following configuration:
   - `base_url`: "https://openrouter.ai/api/v1"
   - `api_key`: Your OpenRouter API key
   - `model`: Any model available on OpenRouter (e.g., "anthropic/claude-sonnet-4-6", "meta-llama/llama-3.3-70b-instruct", etc.)

This gives you access to models from:

- Anthropic (Claude models)
- Meta (Llama models)
- Mistral
- Google (Gemini models)
- And many more

Example:

```python
provider = OpenAIProvider(
    model="mistralai/mistral-large-2512",
    api_key="your_openrouter_key",
    base_url="https://openrouter.ai/api/v1"
)
```

Alternatively, `LiteLLMProvider` (recommended) reaches OpenRouter models directly with an `OPENROUTER_API_KEY` environment variable and an `openrouter/` model prefix, e.g. `LiteLLMProvider(model="openrouter/openai/gpt-4o-mini")`.

## GoogleProvider

### Description

Implementation for Google's Gemini API, supporting the Gemini 3.x and 2.5 models (e.g., `gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-2.5-pro`). This provider enables direct integration with Google's AI models, handling both text and multimodal inputs, as well as structured JSON responses.

### Class Definition

```python
class GoogleProvider(BaseProvider):
    provider: str = "Google"
    api_key: Optional[str] = None
    model: str = "gemini-3.8-flash"
    response_format: Optional[Dict[str, Any]] = None
    response_format_class: Optional[Any] = None
    last_response: Optional[Any] = None
```

### Key Features

- Native Gemini model support via Google's official `genai` library
- Automatic API key handling from environment variables (`GEMINI_API_KEY`)
- Support for structured JSON responses
- Processing of text and image inputs (multimodal capabilities)
- Cost calculation from the token usage Gemini reports, including thinking tokens
- Gemini's `thinking_config` can be passed through `model_args` (see [Model compatibility](#model-compatibility))
- Comprehensive error handling with detailed error messages

### Usage Example

```python
from lattereview.providers import GoogleProvider

# Initialize with the default model (gemini-3.8-flash)
provider = GoogleProvider()

# Or specify a different model
provider = GoogleProvider(model="gemini-3.5-flash-lite")

# Get a text response
response, cost = await provider.get_response("What is the capital of France?")

# Get a response with an image
response, cost = await provider.get_response("What's in this image?", ["path/to/image.jpg"])

# Get a structured JSON response
provider.set_response_format({"name": str, "population": int, "landmarks": [str]})
response, cost = await provider.get_json_response("Give me information about Paris.")
```

### Generation Parameters

Reviewers pass their `model_args` to the provider. `GoogleProvider` copies `temperature`, `top_p`, `top_k`, `max_output_tokens`, `safety_settings` and `thinking_config` into Gemini's generation config and ignores other keys:

```python
from lattereview.agents import ScoringReviewer

reviewer = ScoringReviewer(
    provider=GoogleProvider(model="gemini-3.8-flash"),
    name="Gemini",
    scoring_task="Does the study use deep learning?",
    model_args={"thinking_config": {"thinking_level": "low"}},  # Gemini 2.5: {"thinking_budget": 0}
)
```

### Response Format

The GoogleProvider supports structured JSON responses through the `set_response_format` method. It accepts either a Pydantic model or a Python dictionary that specifies the expected schema:

```python
# Using a dictionary schema
provider.set_response_format({"name": str, "age": int, "hobbies": [str]})

# Using a Pydantic model
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    hobbies: List[str]

provider.set_response_format(Person)
```

## OllamaProvider

### Description

Implementation for local Ollama models, supporting both chat and streaming responses and image inputs. The default model, `qwen3.8:27b`, is a 27B multimodal reasoning model (an 18 GB download) that runs well on an Apple Silicon Mac with 32 GB of memory. Pull it first with `ollama pull qwen3.8:27b`, or pass any other model you have pulled.

### Class Definition

```python
class OllamaProvider(BaseProvider):
    provider: str = "Ollama"
    client: Optional[AsyncClient] = None
    model: str = "qwen3.8:27b"
    response_format_class: Optional[Any] = None
    invalid_keywords: List[str] = ["temperature", "max_tokens"]
    host: str = "http://localhost:11434"
```

### Key Features

- Local model support
- Streaming capability
- Free cost tracking (local models)
- Image input processing
- Structured JSON output constrained to the reviewer's schema
- `model_args` mapping: `reasoning_effort` becomes Ollama's `think` setting (`"none"` turns thinking off; `"low"`, `"medium"` and `"high"` set its level), keywords that `AsyncClient.chat` accepts (`think`, `format`, `options`, `keep_alive`, ...) are passed as they are, and other keys (e.g., `top_p`, `seed`, `num_ctx`) go into the model `options`. `temperature` and `max_tokens` are ignored (see `invalid_keywords`).
- Connection management

### Usage Example

```python
from lattereview.providers import OllamaProvider

# Initialize provider
provider = OllamaProvider(
    model="qwen3.8:27b",
    host="http://localhost:11434"
)

# Get response
response, cost = await provider.get_response("What is the capital of the country shown in this map?", ["path/to/image1.png"])

# Get JSON response with schema
provider.set_response_format({"answer": str, "confidence": float})
response, cost = await provider.get_json_response("What is the capital of France?", [])

# Stream response
async for chunk in await provider.get_response("Tell me a story", [], stream=True):
    print(chunk, end="")

# Clean up
await provider.close()
```

## LiteLLMProvider

### Description

A unified provider implementation using LiteLLM, enabling access to multiple LLM providers through a single interface. It now supports text and image-based interactions.

### Class Definition

```python
class LiteLLMProvider(BaseProvider):
    provider: str = "LiteLLM"
    model: str = "gpt-6-luna"
    custom_llm_provider: Optional[str] = None
    response_format_class: Optional[Any] = None
```

### Key Features

- Support for multiple model providers
- JSON schema validation
- Image input handling
- Cost tracking integration
- Tool calls handling

### Usage Example

```python
from lattereview.providers import LiteLLMProvider

# Initialize with different models
provider = LiteLLMProvider(model="gpt-6-luna")
provider = LiteLLMProvider(model="anthropic/claude-sonnet-5")  # Needs ANTHROPIC_API_KEY
provider = LiteLLMProvider(model="gemini/gemini-3.8-flash")  # Needs GEMINI_API_KEY

# Get response
response, cost = await provider.get_response("What is the capital of the country shown in this map?", ["path/to/image1.png"])

# Get JSON response with schema
provider.set_response_format({"answer": str, "confidence": float})
response, cost = await provider.get_json_response("What is the capital of France?", [])
```

## SystemOneProvider

### Description

`SystemOneProvider` (new in v1.3.0) talks to System One decision models over the `/v1/systemone` protocol. Unlike the LLM providers above, it does not send prompts or parse JSON text: it sends a text **state** and typed **questions** in one request and returns a probability for every allowed answer. It is not a `BaseProvider` subclass, and it is used by `DecisionReviewer`, `DecisionTitleAbstractReviewer` and `DecisionScoringReviewer`. See [Decision Models](../decision_models.md) for the concepts, evaluation results and self-hosting.

### Class Definition

```python
class SystemOneProvider(pydantic.BaseModel):
    backend: Literal["typesafe", "openrouter"] = "typesafe"
    base_url: Optional[str] = None                 # any other /v1/systemone server; overrides backend
    api_key: Optional[str] = None                  # else TYPESAFE_API_KEY / OPENROUTER_API_KEY; optional for base_url
    model: Optional[str] = None                    # else "jev-latest" / "~typesafe/jev-latest"; the server default for base_url
    timeout: float = 60.0
    max_retries: int = 3                           # retries 429, 529, 5xx and timeouts, honoring Retry-After
    requests_per_minute: Optional[float] = None    # 1,000 for the named backends, unlimited for base_url, 0 = off
    input_price_per_million: Optional[float] = None  # 0.042 for the named backends, 0 for base_url

    async def decide(self, state, questions) -> DecisionResult: ...
    async def aclose(self) -> None: ...
```

### Questions and answers

- `Noul(instructions, true=None, false=None)`: a yes/no question.
- `Choice(instructions, options)`: `options` is a dict of option → description, or a list of option names.
- `Score(instructions, levels)`: 2-10 ordered levels, lowest first.

`instructions`, option descriptions and levels may also be JSON structures. The three classes are exported from `lattereview.providers`.

`decide()` returns a `DecisionResult` with `answers` (question ID → `Answer`), `model` (as reported by the server), `input_tokens`, `cost` (reported by OpenRouter, otherwise input tokens × price) and `raw` (the untouched response). Each `Answer` is normalized the same way for every backend:

| Field | noul | choice | score |
| --- | --- | --- | --- |
| `value` | probability of yes | chosen option | expected level (0-based, may be fractional) |
| `label` | None | chosen option | label of the most likely level |
| `level` | None | None | most likely level (0-based) |
| `probabilities` | None | option → probability | level label → probability, in level order |
| `confidence` | None (the probability itself shows certainty) | as reported, else None | as reported, else None |

### Usage Example

```python
from lattereview.providers import SystemOneProvider, Noul, Choice, Score

provider = SystemOneProvider()                                   # TypeSafe; needs TYPESAFE_API_KEY
provider = SystemOneProvider(backend="openrouter")               # needs OPENROUTER_API_KEY
provider = SystemOneProvider(base_url="http://localhost:3000")   # e.g., a local OpenJev server

result = await provider.decide(
    "Title: Deep learning for pneumonia detection on chest radiographs.\nAbstract: ...",
    {
        "deep_learning": Noul("Does the study use deep learning?"),
        "modality": Choice("Which imaging modality is studied?", ["CT", "MRI", "X-ray"]),
        "validation": Score("How strong is the validation?", ["none", "internal only", "external"]),
    },
)
result.answers["modality"].value        # "X-ray"
result.answers["validation"].label      # "external"
result.cost, result.input_tokens
```

Ask all questions about an item in one request: the state is billed once, and each extra question adds only a few tokens.

Errors raise `SystemOneResponseError` (a `ResponseError`) with the server's message, the HTTP `status_code`, and `retryable` (False for errors a retry cannot fix, such as 401 or 422). A missing API key raises `ClientCreationError`.

## Error Handling

Common error scenarios:

- API key errors (missing or invalid keys)
- Unsupported model configurations
- Models not supporting structured outputs or JSON responses
- Invalid image file paths

## Best Practices

1. For all online APIs, prefer using LiteLLMProvider class as it provides unified access and error handling.
2. For local APIs, use OllamaProvider directly (rather than through LiteLLMProvider) for better performance and control.
3. Set API keys at the environment level using the python-dotenv package and .env files for better security.

## Limitations

- Requires async/await syntax for all operations
- Depends on external LLM providers' availability and stability
- Rate limits and quotas depend on provider capabilities
