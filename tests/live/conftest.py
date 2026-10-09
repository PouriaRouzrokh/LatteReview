"""Live tests make real API calls. They run only with `-m live` and skip backends whose key is not set.

Keys are read from the repo's .env (TYPESAFE_API_KEY, OPENROUTER_API_KEY, PERPLEXITY_API_KEY, OPENAI_API_KEY). Set
SYSTEMONE_LOCAL_URL (e.g. http://localhost:3000) to include a local server.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from lattereview.providers import SystemOneProvider

load_dotenv(Path(__file__).parents[2] / ".env")

BACKENDS = {
    "typesafe": ("TYPESAFE_API_KEY", lambda: SystemOneProvider(backend="typesafe")),
    "openrouter": ("OPENROUTER_API_KEY", lambda: SystemOneProvider(backend="openrouter")),
    "openrouter-perplexity": (
        "OPENROUTER_API_KEY",
        lambda: SystemOneProvider(backend="openrouter", model="perplexity/pplx-decider-v1.1-27b"),
    ),
    "perplexity": ("PERPLEXITY_API_KEY", lambda: SystemOneProvider(backend="perplexity")),
    "openai": ("OPENAI_API_KEY", lambda: SystemOneProvider(backend="openai")),
    "local": ("SYSTEMONE_LOCAL_URL", lambda: SystemOneProvider(base_url=os.environ["SYSTEMONE_LOCAL_URL"])),
}


@pytest.fixture(params=list(BACKENDS))
def live_provider(request):
    env_var, factory = BACKENDS[request.param]
    if not os.getenv(env_var):
        pytest.skip(f"{env_var} is not set")
    return factory()
