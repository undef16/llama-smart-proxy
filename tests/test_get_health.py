from unittest.mock import MagicMock

import pytest

from src.entities.model import Model
from src.entities.server import Server
from src.use_cases.get_health import GetHealth


class TestGetHealth:
    @pytest.fixture
    def mock_model_repository(self):
        return MagicMock()

    @pytest.fixture
    def use_case(self, mock_model_repository):
        return GetHealth(mock_model_repository)

    def test_execute_returns_health_dict_with_servers(self, use_case, mock_model_repository):
        # Arrange
        models = [
            Model(id="model1", repo="user/repo1", backend="llama.cpp"),
            Model(id="model2", repo="user/repo2", backend="llama.cpp"),
        ]
        servers = [
            Server(id="server1", host="localhost", port=8080, model_id="model1", status="running", process=123),
            Server(id="server2", host="localhost", port=8081, model_id="model2", status="stopped"),
        ]

        mock_model_repository.get_all_models.return_value = models
        mock_model_repository.get_servers_for_model.side_effect = lambda model_id: [
            s for s in servers if s.model_id == model_id
        ]

        # Act
        result = use_case.execute()

        # Assert
        mock_model_repository.get_all_models.assert_called_once()
        mock_model_repository.get_servers_for_model.assert_any_call("model1")
        mock_model_repository.get_servers_for_model.assert_any_call("model2")
        expected = {"servers": [server.model_dump() for server in servers]}
        assert result == expected

    def test_execute_with_no_models_returns_empty_servers(self, use_case, mock_model_repository):
        # Arrange
        mock_model_repository.get_all_models.return_value = []
        mock_model_repository.get_servers_for_model.return_value = []

        # Act
        result = use_case.execute()

        # Assert
        mock_model_repository.get_all_models.assert_called_once()
        assert result == {"servers": []}
