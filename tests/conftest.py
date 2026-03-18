"""Shared pytest configuration and fixtures."""

import os

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "live: mark test as requiring live API calls")


@pytest.fixture
def env_keys():
    """Load API keys from .env, skip if dotenv not available."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
