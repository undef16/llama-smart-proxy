from typing import Any, Optional, Protocol
from .common_imports import BaseModel, Field, Dict, List


class Message(BaseModel):
    role: str = Field(..., description="The role of the message author (e.g., 'user', 'assistant', 'system')")
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="The model to use for the chat completion")
    messages: List[Message] = Field(..., description="A list of messages comprising the conversation so far")
    temperature: Optional[float] = Field(None, description="Controls randomness in the output")
    max_tokens: Optional[int] = Field(None, description="The maximum number of tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    # Add other optional fields as needed


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class AgentConfig(BaseModel):
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