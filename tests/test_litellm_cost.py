"""Unit tests for LiteLLMProvider's cost bookkeeping and Perplexity routing (no network)."""

from types import SimpleNamespace

import pytest

from lattereview.providers import LiteLLMProvider


def response(cost):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=250, completion_tokens=50, cost=cost))


@pytest.mark.parametrize(
    "cost, expected",
    [
        (0.0053, 0.0053),  # OpenRouter reports a number, including Sonar's per-request search fee
        ({"request_cost": 0.005, "total_cost": 0.00529}, 0.00529),  # Perplexity reports a breakdown
        (0, None),  # e.g., OpenRouter with your own provider key: fall back to token prices
        (None, None),
        (True, None),
        ({"request_cost": 0.005}, None),
    ],
)
def test_reported_cost(cost, expected):
    assert LiteLLMProvider._reported_cost(response(cost)) == expected


def test_reported_cost_wins_over_token_prices():
    provider = LiteLLMProvider(model="openrouter/perplexity/sonar")
    assert provider._safe_completion_cost(response(0.0053)) == 0.0053


@pytest.mark.parametrize(
    "model, routed",
    [
        ("perplexity/sonar", "perplexity/responses/perplexity/sonar"),
        ("perplexity/openai/gpt-6-luna", "perplexity/responses/openai/gpt-6-luna"),
        ("perplexity/responses/perplexity/sonar", "perplexity/responses/perplexity/sonar"),
        ("openrouter/perplexity/sonar", "openrouter/perplexity/sonar"),
        ("gpt-6-luna", "gpt-6-luna"),
    ],
)
def test_perplexity_models_go_through_the_agent_api(model, routed):
    name, kwargs = LiteLLMProvider(model=model)._route({"temperature": 0.1})
    assert name == routed
    assert kwargs.get("num_retries") == (5 if model.startswith("perplexity/") and "responses" not in model else None)
    assert LiteLLMProvider(model="perplexity/sonar")._route({"num_retries": 1})[1]["num_retries"] == 1
