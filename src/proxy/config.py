import json
from pathlib import Path
from typing import Optional
from .common_imports import BaseModel, Field, Dict, List


class ServerPoolConfig(BaseModel):
    """Configuration for the server pool.

    Attributes:
        size: Number of servers in the pool.
        host: Host for the servers.
        port_start: Starting port number for servers.
    """
    size: int = Field(..., gt=0, description="Number of servers in the pool")
    host: str = Field("localhost", description="Host for the servers")
    port_start: int = Field(8001, description="Starting port number for servers")


class ModelConfig(BaseModel):
    """Configuration for a model.

    Attributes:
        repo: Repository for the model.
        variant: Model variant.
    """
    repo: str = Field(..., description="Repository for the model")
    variant: str = Field(..., description="Model variant")


class MessageConfig(BaseModel):
    """Configuration for a message.

    Attributes:
        role: Role of the message sender.
        content: Content of the message.
    """
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")


class SimulationConfig(BaseModel):
    """Configuration for simulation.

    Attributes:
        server_url: URL of the server.
        health_endpoint: Health check endpoint.
        chat_endpoint: Chat completions endpoint.
        wait_timeout: Timeout for waiting for server.
        request_timeout: Timeout for requests.
        terminate_timeout: Timeout for terminating server.
        model: Model to use for simulation.
        messages: Messages for the simulation.
    """
    server_url: str = Field(..., description="URL of the server")
    health_endpoint: str = Field(..., description="Health check endpoint")
    chat_endpoint: str = Field(..., description="Chat completions endpoint")
    wait_timeout: int = Field(30, description="Timeout for waiting for server")
    request_timeout: int = Field(300, description="Timeout for requests")
    terminate_timeout: int = Field(10, description="Timeout for terminating server")
    model: str = Field(..., description="Model to use for simulation")
    messages: List[MessageConfig] = Field(default_factory=list, description="Messages for the simulation")
class Config(BaseModel):
    """Main configuration class.

    Attributes:
        server_pool: Configuration for the server pool.
        models: Mapping of model names to configurations.
        agents: List of available agent plugins.
        simulation: Simulation configuration.
    """
    server_pool: ServerPoolConfig
    models: Dict[str, ModelConfig] = Field(default_factory=dict, description="Mapping of model names to configurations")
    agents: List[str] = Field(default_factory=list, description="List of available agent plugins")
    simulation: Optional[SimulationConfig] = Field(default=None, description="Simulation configuration")



    @classmethod
    def load(cls, config_path: str = "config.json") -> "Config":
        """Load and validate configuration from JSON file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(path, 'r') as f:
            data = json.load(f)

        return cls(**data)