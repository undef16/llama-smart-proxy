import json
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field


class ServerPoolConfig(BaseModel):
    """Configuration for the server pool.

    Attributes:
        size: Number of servers in the pool.
        host: Host for the servers.
        port_start: Starting port number for servers.
        gpu_layers: Number of layers to offload to GPU (0 = CPU only).
        request_timeout: Timeout for requests to server pool in seconds.
    """

    size: int = Field(..., gt=0, description="Number of servers in the pool")
    host: str = Field("localhost", description="Host for the servers")
    port_start: int = Field(8001, description="Starting port number for servers")
    gpu_layers: int = Field(0, ge=0, description="Number of layers to offload to GPU (0 = CPU only)")
    request_timeout: int = Field(300, description="Timeout for requests to server pool in seconds")


class ServerConfig(BaseModel):
    """Configuration for the main server.

    Attributes:
        host: Host for the main server.
        port: Port for the main server.
    """

    host: str = Field("0.0.0.0", description="Host for the main server")
    port: int = Field(8000, description="Port for the main server")


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


class OllamaConfig(BaseModel):
    """Configuration for Ollama backend.

    Attributes:
        host: Host for the Ollama server.
        port: Port for the Ollama server.
        timeout: Timeout for Ollama requests.
    """

    host: str = Field("localhost", description="Host for the Ollama server")
    port: int = Field(11434, description="Port for the Ollama server")
    timeout: float = Field(30.0, description="Timeout for Ollama requests")


class SimulationConfig(BaseModel):
    """Configuration for simulation.

    Attributes:
        enable_remote: Whether to use a remote server instead of starting local.
        host: Host for the remote server.
        port: Port for the remote server.
        server_url: URL of the server (deprecated, use host/port instead).
        health_endpoint: Health check endpoint.
        chat_endpoint: Chat completions endpoint.
        wait_timeout: Timeout for waiting for server.
        request_timeout: Timeout for requests.
        terminate_timeout: Timeout for terminating server.
        model: Model to use for simulation.
        messages: Messages for the simulation.
        ollama_models: List of Ollama models available for simulation.
        llama_cpp_models: Mapping of model names to configurations for simulation.
    """

    enable_remote: bool = Field(False, description="Whether to use a remote server instead of starting local")
    host: str = Field("localhost", description="Host for the remote server")
    port: int = Field(8000, description="Port for the remote server")
    server_url: str | None = Field(None, description="URL of the server (deprecated, use host/port instead)")
    health_endpoint: str = Field(..., description="Health check endpoint")
    chat_endpoint: str = Field(..., description="Chat completions endpoint")
    wait_timeout: int = Field(30, description="Timeout for waiting for server")
    request_timeout: int = Field(300, description="Timeout for requests")
    terminate_timeout: int = Field(10, description="Timeout for terminating server")
    max_tokens: int = Field(2048, description="Maximum tokens for simulation requests")
    model: str = Field(..., description="Model to use for simulation")
    messages: List[MessageConfig] = Field(default_factory=list, description="Messages for the simulation")
    ollama_models: List[str] = Field(default_factory=list, description="List of Ollama models available for simulation")
    llama_cpp_models: Dict[str, ModelConfig] = Field(default_factory=dict, description="Mapping of model names to configurations for simulation")


class GPUConfig(BaseModel):
    """Configuration for GPU settings.

    Attributes:
        enabled: Whether GPU monitoring and allocation is enabled.
        enable_gpu_monitoring: Whether GPU monitoring is enabled.
        allocation_strategy: Strategy for GPU allocation ("single-gpu-preferred" or "distribute").
        gpu_allocation_strategy: Strategy for GPU allocation (alias for allocation_strategy).
        monitoring_interval: Interval for GPU monitoring in seconds.
        cpu_fallback: Whether to fallback to CPU when GPU is not available.
        kv_offload: Whether KV cache offloading to CPU is enabled.
        cache_type_k: KV cache quantization type for K.
        cache_type_v: KV cache quantization type for V.
        repack: Whether weight repacking is enabled.
        no_host: Whether to bypass host buffers.
        activation_overhead_factor: Overhead factor for activation memory.
    """

    enabled: bool = Field(True, description="Whether GPU monitoring and allocation is enabled")
    enable_gpu_monitoring: bool = Field(True, description="Whether GPU monitoring is enabled")
    allocation_strategy: str = Field("single-gpu-preferred", description="Strategy for GPU allocation ('single-gpu-preferred' or 'distribute')")
    gpu_allocation_strategy: str | None = Field(None, description="Strategy for GPU allocation (overrides allocation_strategy if set)")
    monitoring_interval: float = Field(5.0, description="Interval for GPU monitoring in seconds")
    cpu_fallback: bool = Field(True, description="Whether to fallback to CPU when GPU is not available")
    kv_offload: bool = Field(True, description="Whether KV cache offloading to CPU is enabled")
    cache_type_k: str = Field("f16", description="KV cache quantization type for K")
    cache_type_v: str = Field("f16", description="KV cache quantization type for V")
    repack: bool = Field(True, description="Whether weight repacking is enabled")
    no_host: bool = Field(False, description="Whether to bypass host buffers")
    activation_overhead_factor: float = Field(0.25, description="Overhead factor for activation memory")

    @property
    def effective_allocation_strategy(self) -> str:
        """Get the effective allocation strategy, preferring gpu_allocation_strategy if set."""
        return self.gpu_allocation_strategy or self.allocation_strategy


class Config(BaseModel):
    """Main configuration class.

    Attributes:
        backend: Backend to use ('llama.cpp' or 'ollama').
        server_pool: Configuration for the server pool.
        ollama: Configuration for Ollama backend.
        server: Configuration for the main server.
        models: Mapping of model names to configurations (deprecated, use simulation.llama_cpp_models).
        agents: List of available agent plugins.
        simulation: Simulation configuration.
        gpu: GPU configuration settings.
    """

    backend: str = Field(..., description="Backend to use ('llama.cpp' or 'ollama')")
    server_pool: ServerPoolConfig
    ollama: OllamaConfig | None = Field(default=None, description="Configuration for Ollama backend")
    server: ServerConfig
    models: Dict[str, ModelConfig] = Field(default_factory=dict, description="Mapping of model names to configurations (deprecated, use simulation.llama_cpp_models)")
    agents: List[str] = Field(default_factory=list, description="List of available agent plugins")
    simulation: SimulationConfig | None = Field(default=None, description="Simulation configuration")
    gpu: GPUConfig | None = Field(default=None, description="GPU configuration settings")

    @property
    def effective_models(self) -> Dict[str, ModelConfig]:
        """Get the effective models configuration, preferring simulation.llama_cpp_models if available."""
        if self.simulation and self.simulation.llama_cpp_models:
            return self.simulation.llama_cpp_models
        return self.models

    @classmethod
    def load(cls, config_path: str = "config.json") -> "Config":
        """Load and validate configuration from JSON file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(path) as f:
            data = json.load(f)

        return cls(**data)