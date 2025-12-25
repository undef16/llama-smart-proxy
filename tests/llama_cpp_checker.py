import requests
from typing import Optional

from src.shared.protocols import BackendCheckerProtocol


class LlamaCppChecker(BackendCheckerProtocol):
    """Independent checker for llama.cpp backend availability."""

    def __init__(self, config: dict, host: str = "localhost", port_start: int = 8080, pool_size: int = 1):
        self.config = config
        self.host = host
        self.port_start = port_start
        self.pool_size = pool_size

    def check_server_availability(self, port: int) -> bool:
        """Check if a llama.cpp server is healthy on the given port."""
        try:
            response = requests.get(f"http://{self.host}:{port}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            return False

    def check_availability(self, model: Optional[str] = None) -> bool:
        """Check if servers can be started on demand."""
        # Since servers are started on demand by ServerPool, assume available
        return True

    def get_forwarding_endpoints(self) -> list[dict]:
        """Get list of endpoints that can be forwarded for llama.cpp."""
        return [
            {"endpoint": "/tokenize", "data_key": "model"},
            {"endpoint": "/detokenize", "data_key": "model"}
        ]

    def get_simulation_model(self) -> str:
        """Get the model to use for simulation."""
        model_key = self.config.get("simulation", {}).get("model", "")
        models = self.config.get("simulation", {}).get("llama_cpp_models", {})
        if model_key and model_key in models:
            # Use the specified model from simulation.model
            repo = models[model_key]["repo"]
            variant = models[model_key]["variant"]
            return f"{repo}:{variant}"
        elif models:
            # Fallback: use the first available model
            model_key = list(models.keys())[0]
            repo = models[model_key]["repo"]
            variant = models[model_key]["variant"]
            return f"{repo}:{variant}"
        raise ValueError("No models configured for llama.cpp")