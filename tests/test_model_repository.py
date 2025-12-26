import pytest

from src.entities.model import Model
from src.entities.server import Server
from src.frameworks_drivers.model_repository import ModelRepository


class TestModelRepository:
    @pytest.fixture
    def repository(self):
        return ModelRepository()

    def test_get_model_returns_model_when_exists(self, repository):
        # Arrange
        model = Model(id="model1", repo="user/repo", backend="llama.cpp")
        repository.models["model1"] = model

        # Act
        result = repository.get_model("model1")

        # Assert
        assert result == model

    def test_get_model_raises_keyerror_when_not_exists(self, repository):
        # Act & Assert
        with pytest.raises(KeyError):
            repository.get_model("nonexistent")

    def test_get_all_models_returns_list_of_all_models(self, repository):
        # Arrange
        model1 = Model(id="model1", repo="user/repo1", backend="llama.cpp")
        model2 = Model(id="model2", repo="user/repo2", backend="ollama")
        repository.models = {"model1": model1, "model2": model2}

        # Act
        result = repository.get_all_models()

        # Assert
        assert result == [model1, model2]

    def test_get_servers_for_model_returns_servers_for_given_model(self, repository):
        # Arrange
        server1 = Server(id="server1", host="localhost", port=8080, model_id="model1", status="running")
        server2 = Server(id="server2", host="localhost", port=8081, model_id="model1", status="stopped")
        server3 = Server(id="server3", host="localhost", port=8082, model_id="model2", status="running")
        repository.servers = [server1, server2, server3]

        # Act
        result = repository.get_servers_for_model("model1")

        # Assert - expect ServerDTOs (dictionaries) instead of Server objects
        expected = [
            {
                "id": "server1",
                "host": "localhost",
                "port": 8080,
                "model_id": "model1",
                "status": "running",
                "process": None,
                "gpu_assignment": None
            },
            {
                "id": "server2",
                "host": "localhost",
                "port": 8081,
                "model_id": "model1",
                "status": "stopped",
                "process": None,
                "gpu_assignment": None
            }
        ]
        assert result == expected

    def test_get_servers_for_model_returns_empty_list_when_no_servers(self, repository):
        # Arrange
        repository.servers = []

        # Act
        result = repository.get_servers_for_model("model1")

        # Assert
        assert result == []

    def test_get_model_creates_dynamic_model_when_not_exists(self, repository):
        # Act
        model = repository.get_model("Qwen/Qwen2.5-0.5B-Instruct-GGUF")

        # Assert
        assert model.id == "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        assert model.repo == "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        assert model.variant == "Q4_K_M"  # Default variant
        assert model.backend == "llama.cpp"
        assert model.parameters == 500_000_000  # 0.5B = 500M parameters

    def test_get_or_create_model_caches_created_models(self, repository):
        # Act
        model1 = repository.get_model("user/test-model")
        model2 = repository.get_model("user/test-model")

        # Assert
        assert model1 is model2
        assert model1.id == "user/test-model"

    def test_get_model_creates_local_gguf_model(self, repository):
        # Act
        model = repository.get_model("model.gguf")

        # Assert
        assert model.id == "model.gguf"
        assert model.repo == "local"
        assert model.variant == "model"
        assert model.backend == "llama.cpp"
