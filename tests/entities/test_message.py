import pytest
from pydantic import ValidationError

from src.entities.message import Message


def test_message_creation_valid():
    message = Message(role="user", content="Hello world")
    assert message.role == "user"
    assert message.content == "Hello world"


def test_message_creation_assistant():
    message = Message(role="assistant", content="Response")
    assert message.role == "assistant"


def test_message_creation_system():
    message = Message(role="system", content="System prompt")
    assert message.role == "system"


def test_message_invalid_role():
    with pytest.raises(ValidationError):
        Message(role="invalid", content="test")


def test_message_empty_content():
    with pytest.raises(ValidationError):
        Message(role="user", content="")
