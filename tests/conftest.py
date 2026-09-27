"""Shared unit-test fixtures."""

import httpx
import pytest

from lattereview.providers import SystemOneProvider

from systemone_fixtures import Recorder


@pytest.fixture
def make_provider(monkeypatch):
    """Return a factory for providers whose HTTP calls go to a Recorder instead of the network."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-typesafe-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    async def no_sleep(_):
        return None

    monkeypatch.setattr("lattereview.providers.system_one_provider.asyncio.sleep", no_sleep)

    def factory(*responses, **kwargs):
        recorder = Recorder(*responses)
        provider = SystemOneProvider(transport=httpx.MockTransport(recorder), **kwargs)
        return provider, recorder

    return factory
