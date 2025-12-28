import pytest
from pydantic import ValidationError

from src.entities.agent import Agent


def test_agent_creation_valid():
    agent = Agent(name="test-agent", enabled=True, hooks=["request", "response"])
    assert agent.name == "test-agent"
    assert agent.enabled is True
    assert agent.hooks == ["request", "response"]


def test_agent_creation_disabled():
    agent = Agent(name="test-agent", enabled=False, hooks=[])
    assert agent.enabled is False
    assert agent.hooks == []


def test_agent_invalid_name_empty():
    with pytest.raises(ValidationError):
        Agent(name="", enabled=True, hooks=[])


def test_agent_invalid_hooks():
    with pytest.raises(ValidationError):
        Agent(name="test-agent", enabled=True, hooks=["invalid_hook"])
