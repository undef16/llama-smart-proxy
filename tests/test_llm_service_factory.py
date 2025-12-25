
import pytest

from src.frameworks_drivers.llama_cpp_service import LlamaCppLLMService
from src.frameworks_drivers.llm_service_factory import LLMServiceFactory
from src.frameworks_drivers.ollama_service import OllamaLLMService


class TestLLMServiceFactory:
    def test_create_llama_cpp_service(self):
        from unittest.mock import Mock
        config = {"backend": "llama.cpp", "server_pool": {"host": "localhost", "port_start": 8080}}

        mock_server_pool = Mock()
        mock_model_resolver = Mock()
        factory = LLMServiceFactory(config, server_pool=mock_server_pool, model_resolver=mock_model_resolver)
        service = factory.create_service()

        assert isinstance(service, LlamaCppLLMService)
        assert service.server_pool == mock_server_pool
        assert service.model_resolver == mock_model_resolver

    def test_create_ollama_service(self):
        config = {"backend": "ollama", "ollama": {"host": "127.0.0.1", "port": 11434}}

        factory = LLMServiceFactory(config)
        service = factory.create_service()

        assert isinstance(service, OllamaLLMService)
        assert service.host == "127.0.0.1"
        assert service.port == 11434

    def test_invalid_backend(self):
        config = {"backend": "invalid"}

        factory = LLMServiceFactory(config)

        with pytest.raises(ValueError, match="Unsupported backend: invalid"):
            factory.create_service()

    def test_missing_backend(self):
        config = {}

        factory = LLMServiceFactory(config)

        with pytest.raises(ValueError, match="Backend not specified in config"):
            factory.create_service()

    def test_llama_cpp_missing_server_pool(self):
        config = {"backend": "llama.cpp"}

        factory = LLMServiceFactory(config)

        with pytest.raises(ValueError, match="ServerPool not provided for llama.cpp service"):
            factory.create_service()

    def test_ollama_missing_ollama_config(self):
        config = {"backend": "ollama"}

        factory = LLMServiceFactory(config)

        with pytest.raises(ValueError, match="ollama config not found"):
            factory.create_service()
