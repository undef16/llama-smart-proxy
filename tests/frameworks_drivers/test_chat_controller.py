from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from src.interface_adapters.chat_controller import ChatController


class TestChatController:
    @pytest.fixture
    def mock_use_case(self):
        return AsyncMock()

    @pytest.fixture
    def controller(self, mock_use_case):
        return ChatController(mock_use_case)

    @pytest.mark.asyncio
    async def test_chat_completions_valid_request_calls_use_case_and_returns_response(self, controller, mock_use_case):
        # Arrange
        request = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}
        expected_response = {"choices": [{"message": {"content": "Response"}}]}

        mock_use_case.execute.return_value = expected_response

        # Act
        result = await controller.chat_completions(request)

        # Assert
        mock_use_case.execute.assert_called_once_with(request)
        assert result == expected_response

    @pytest.mark.asyncio
    async def test_chat_completions_missing_model_raises_http_exception(self, controller, mock_use_case):
        # Arrange
        request = {"messages": [{"role": "user", "content": "Hello"}]}

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await controller.chat_completions(request)

        assert exc_info.value.status_code == 400
        assert "Model must be a non-empty string" in str(exc_info.value.detail)
        mock_use_case.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_completions_empty_model_raises_http_exception(self, controller, mock_use_case):
        # Arrange
        request = {"model": "", "messages": [{"role": "user", "content": "Hello"}]}

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await controller.chat_completions(request)

        assert exc_info.value.status_code == 400
        assert "Model must be a non-empty string" in str(exc_info.value.detail)
        mock_use_case.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_completions_non_string_model_raises_http_exception(self, controller, mock_use_case):
        # Arrange
        request = {"model": 123, "messages": [{"role": "user", "content": "Hello"}]}

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await controller.chat_completions(request)

        assert exc_info.value.status_code == 400
        assert "Model must be a non-empty string" in str(exc_info.value.detail)
        mock_use_case.execute.assert_not_called()
