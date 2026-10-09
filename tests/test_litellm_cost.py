"""Unit tests for LiteLLMProvider's cost bookkeeping (no network)."""

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
