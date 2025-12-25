import ollama
from typing import Optional

from src.shared.protocols import BackendCheckerProtocol


class OllamaChecker(BackendCheckerProtocol):
    """Independent checker for Ollama backend availability."""

    def __init__(self, config: dict, host: str = "localhost", port: int = 11434):
        self.config = config
        self.host = host
        self.port = port

    def check_server_availability(self) -> bool:
        """Check if Ollama server is running."""
        try:
            client = ollama.Client(host=f"http://{self.host}:{self.port}")
            client.list()
            return True
        except Exception:
            return False

    def check_model_availability(self, model: str) -> bool:
        """Check if a specific model is available in Ollama."""
        try:
            client = ollama.Client(host=f"http://{self.host}:{self.port}")
            models_response = client.list()
            available_models = [m.model for m in models_response.models]
            return model in available_models
        except Exception:
            return False

    def check_availability(self, model: Optional[str] = None) -> bool:
        """Check overall availability, optionally for a specific model."""
        if not self.check_server_availability():
            return False
        if model and not self.check_model_availability(model):
            return False
        return True

    def get_forwarding_endpoints(self) -> list[dict]:
        """Get list of endpoints that can be forwarded for Ollama."""
        return [
            {"endpoint": "/api/show", "data_key": "name"},
        ]

    def get_simulation_model(self) -> str:
        """Get the model to use for simulation."""
        ollama_models = self.config.get("ollama", {}).get("models", [])
        if ollama_models:
            return ollama_models[0]
        raise ValueError("No Ollama models configured")