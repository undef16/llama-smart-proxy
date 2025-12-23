from typing import Any, Optional, Protocol
from .common_imports import BaseModel, Field, Dict, List


class Message(BaseModel):
    """Represents a message in a chat conversation.

    Attributes:
        role: The role of the message author (e.g., 'user', 'assistant', 'system').
        content: The content of the message.
    """
    role: str = Field(..., description="The role of the message author (e.g., 'user', 'assistant', 'system')")
    content: str = Field(..., description="The content of the message")


class AgentConfig(BaseModel):
    """Configuration for an agent plugin.

    Attributes:
        description: Description of the agent.
        enabled: Whether the agent is enabled.
        end_point: Endpoint this agent applies to.
    """
    description: str = Field(..., description="Description of the agent")
    enabled: bool = Field(True, description="Whether the agent is enabled")
    end_point: str = Field("/chat/completions", description="Endpoint this agent applies to")


class Agent(Protocol):
    """Protocol for agent plugins."""

    def process_request(self, request: dict) -> dict:
        """Process the request before sending to the model."""
        ...

    def process_response(self, response: dict) -> dict:
        """Process the response after receiving from the model."""
        ...