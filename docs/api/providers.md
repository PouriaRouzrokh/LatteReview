# Providers Module Documentation

This module provides different language model provider implementations for the LatteReview package. It handles interactions with various LLM APIs in a consistent, type-safe manner and now supports image-based inputs in addition to text inputs.

## Overview

The providers module includes:

- `BaseProvider`: Abstract base class defining the provider interface
- `OpenAIProvider`: Implementation for OpenAI API (including GPT models)
- `OllamaProvider`: Implementation for local Ollama models
- `LiteLLMProvider`: Implementation using LiteLLM for unified API access

** You can use any of the models offered by the providers above as far as they support structured outputs. **

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

## OpenAIProvider

### Description

Implementation for OpenAI's API, supporting both OpenAI models and Gemini models through a compatible endpoint. The updated version supports processing text and image inputs.

### Class Definition

```python
class OpenAIProvider(BaseProvider):
    provider: str = "OpenAI"
    api_key: str = None
    client: Optional[openai.AsyncOpenAI] = None
    model: str = "gpt-4o-mini"
    response_format_class: Optional[Any] = None
```

### Key Features

- Automatic API key handling from environment variables
- Support for OpenAI and Gemini models
- JSON response validation
- Image input handling
- Comprehensive error handling

### Usage Example

```python
from lattereview.providers import OpenAIProvider

# Initialize with OpenAI model
provider = OpenAIProvider(model="gpt-4o")

# Initialize with Gemini model
provider = OpenAIProvider(model="gemini/gemini-1.5-flash")

# Get a response
response, cost = await provider.get_response("What is the capital of the country shown in this map?", ["path/to/image.jpg"])

# Get JSON response
provider.set_response_format({"key": str, "value": int})
response, cost = await provider.get_json_response("Format this as JSON", [])
```

## OllamaProvider

### Description

Implementation for local Ollama models, supporting both chat and streaming responses. The updated version handles image inputs for advanced tasks.

### Class Definition

```python
class OllamaProvider(BaseProvider):
    provider: str = "Ollama"
    client: Optional[AsyncClient] = None
    model: str = "llama3.2-vision:latest"
    response_format_class: Optional[Any] = None
    invalid_keywords: List[str] = ["temperature", "max_tokens"]
    host: str = "http://localhost:11434"
```

### Key Features

- Local model support
- Streaming capability
- Free cost tracking (local models)
- Image input processing
- Connection management

### Usage Example

```python
from lattereview.providers import OllamaProvider

# Initialize provider
provider = OllamaProvider(
    model="llama3.2-vision:latest",
    host="http://localhost:11434"
)

# Get response
response, cost = await provider.get_response("What is the capital of the country shown in this map?", ["path/to/image1.png"])

# Get JSON response with schema
provider.set_response_format({"answer": str, "confidence": float})
response, cost = await provider.get_json_response("What is the capital of France?", [])

# Stream response
async for chunk in provider.get_response("Tell me a story", [], stream=True):
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
    model: str = "gpt-4o-mini"
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
provider = LiteLLMProvider(model="gpt-4o-mini")

# Get response
response, cost = await provider.get_response("What is the capital of the country shown in this map?", ["path/to/image1.png"])

# Get JSON response with schema
provider.set_response_format({"answer": str, "confidence": float})
response, cost = await provider.get_json_response("What is the capital of France?", [])
```

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
