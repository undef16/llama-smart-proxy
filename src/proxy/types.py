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


class ChatCompletionRequest(BaseModel):
    """Represents a request for chat completion.

    Attributes:
        model: The model to use for the chat completion.
        messages: A list of messages comprising the conversation so far.
        temperature: Controls randomness in the output.
        max_tokens: The maximum number of tokens to generate.
        stream: Whether to stream the response.
    """
    model: str = Field(..., description="The model to use for the chat completion")
    messages: List[Message] = Field(..., description="A list of messages comprising the conversation so far")
    temperature: Optional[float] = Field(None, description="Controls randomness in the output")
    max_tokens: Optional[int] = Field(None, description="The maximum number of tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    # Add other optional fields as needed


class ChatCompletionChoice(BaseModel):
    """Represents a single choice in a chat completion response.

    Attributes:
        index: The index of the choice.
        message: The message content of the choice.
        finish_reason: The reason the generation finished.
    """
    index: int
    message: Message
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    """Represents token usage statistics for a chat completion.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total number of tokens used.
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Represents the response from a chat completion request.

    Attributes:
        id: Unique identifier for the response.
        object: The object type, typically 'chat.completion'.
        created: Timestamp of when the response was created.
        model: The model used for the completion.
        choices: List of completion choices.
        usage: Token usage statistics.
    """
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


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

    def process_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Process the request before sending to the model."""
        ...

    def process_response(self, response: ChatCompletionResponse) -> ChatCompletionResponse:
        """Process the response after receiving from the model."""
        ...