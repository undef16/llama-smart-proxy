import json
from pathlib import Path
from .common_imports import BaseModel, Field, Dict, List


class ServerPoolConfig(BaseModel):
    size: int = Field(..., gt=0, description="Number of servers in the pool")
    host: str = Field("localhost", description="Host for the servers")
    port_start: int = Field(8001, description="Starting port number for servers")


class ModelConfig(BaseModel):
    repo: str = Field(..., description="Repository for the model")
    variant: str = Field(..., description="Model variant")


class Config(BaseModel):
    server_pool: ServerPoolConfig
    models: Dict[str, ModelConfig] = Field(default_factory=dict, description="Mapping of model names to configurations")
    agents: List[str] = Field(default_factory=list, description="List of available agent plugins")

    @classmethod
    def load(cls, config_path: str = "config.json") -> "Config":
        """Load and validate configuration from JSON file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(path, 'r') as f:
            data = json.load(f)

        return cls(**data)