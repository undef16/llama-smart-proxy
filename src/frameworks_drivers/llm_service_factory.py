import requests

from src.frameworks_drivers.llama_cpp_service import LlamaCppLLMService
from src.frameworks_drivers.model_resolver import ModelResolver
from src.frameworks_drivers.ollama_service import OllamaLLMService
from src.frameworks_drivers.server_pool import ServerPool
from src.shared.protocols import LLMServiceProtocol


class LLMServiceFactory:
    def __init__(self, config: dict, server_pool: ServerPool | None = None, model_resolver: ModelResolver | None = None):
        self.config = config
        self.server_pool = server_pool
        self.model_resolver = model_resolver

    def create_service(self) -> LLMServiceProtocol:
        backend = self.config.get("backend")
        if not backend:
            raise ValueError("Backend not specified in config")

        if backend == "llama.cpp":
            return self._create_llama_cpp_service()
        if backend == "ollama":
            return self._create_ollama_service()
        raise ValueError(f"Unsupported backend: {backend}")

    def _create_llama_cpp_service(self) -> LlamaCppLLMService:
        if not self.server_pool:
            raise ValueError("ServerPool not provided for llama.cpp service")
        if not self.model_resolver:
            raise ValueError("ModelResolver not provided for llama.cpp service")

        timeout = self.config.get("server_pool", {}).get("request_timeout", 30.0)

        return LlamaCppLLMService(server_pool=self.server_pool, model_resolver=self.model_resolver, timeout=timeout)

    def _create_ollama_service(self) -> OllamaLLMService:
        ollama_config = self.config.get("ollama")
        if not ollama_config:
            raise ValueError("ollama config not found")

        host = ollama_config.get("host", "localhost")
        port = ollama_config.get("port", 11434)
        timeout = ollama_config.get("timeout", 30.0)

        return OllamaLLMService(host=host, port=port, timeout=timeout)
