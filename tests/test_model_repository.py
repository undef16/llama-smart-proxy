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

        # Assert
        assert result == [server1, server2]

    def test_get_servers_for_model_returns_empty_list_when_no_servers(self, repository):
        # Arrange
        repository.servers = []

        # Act
        result = repository.get_servers_for_model("model1")

        # Assert
        assert result == []
