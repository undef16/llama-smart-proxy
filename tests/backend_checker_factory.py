from src.shared.protocols import BackendCheckerProtocol
from tests.llama_cpp_checker import LlamaCppChecker
from tests.ollama_checker import OllamaChecker


class BackendCheckerFactory:
    def __init__(self, config: dict):
        self.config = config

    def create_checker(self) -> BackendCheckerProtocol:
        backend = self.config.get("backend")
        if not backend:
            raise ValueError("Backend not specified in config")

        if backend == "llama.cpp":
            return self._create_llama_cpp_checker()
        if backend == "ollama":
            return self._create_ollama_checker()
        raise ValueError(f"Unsupported backend: {backend}")

    def _create_llama_cpp_checker(self) -> LlamaCppChecker:
        server_pool = self.config.get("server_pool")
        if not server_pool:
            raise ValueError("server_pool not found in config")

        host = server_pool.get("host", "localhost")
        port_start = server_pool.get("port_start", 8080)
        pool_size = server_pool.get("size", 1)

        return LlamaCppChecker(config=self.config, host=host, port_start=port_start, pool_size=pool_size)

    def _create_ollama_checker(self) -> OllamaChecker:
        ollama_config = self.config.get("ollama")
        if not ollama_config:
            raise ValueError("ollama config not found")

        host = ollama_config.get("host", "localhost")
        port = ollama_config.get("port", 11434)

        return OllamaChecker(config=self.config, host=host, port=port)