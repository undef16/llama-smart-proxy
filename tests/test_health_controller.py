from unittest.mock import MagicMock

import pytest

from src.interface_adapters.health_controller import HealthController


class TestHealthController:
    @pytest.fixture
    def mock_use_case(self):
        return MagicMock()

    @pytest.fixture
    def controller(self, mock_use_case):
        return HealthController(mock_use_case)

    def test_health_calls_use_case_and_returns_response(self, controller, mock_use_case):
        # Arrange
        expected_response = {"servers": [{"id": "server1", "status": "running"}]}
        mock_use_case.execute.return_value = expected_response

        # Act
        result = controller.health()

        # Assert
        mock_use_case.execute.assert_called_once()
        assert result == expected_response
