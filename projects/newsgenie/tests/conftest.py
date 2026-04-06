"""Shared pytest fixtures for the NewsGenie test suite."""

import pytest


@pytest.fixture
def sample_session_id() -> str:
    """A reusable session UUID for tests that require one."""
    return "test-session-1234"
