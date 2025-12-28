from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.frameworks_drivers.llama_cpp_service import LlamaCppLLMService


class TestLlamaCppLLMService:
    @pytest.fixture
    def service(self):
        from unittest.mock import Mock, AsyncMock
        mock_server = Mock()
        mock_server.host = "localhost"
        mock_server.port = 8080
        mock_server_pool = Mock()
        mock_server_pool.get_server_for_model = AsyncMock(return_value=mock_server)
        mock_model_resolver = Mock()
        mock_model_resolver.resolve.return_value = ("test/repo", "Q4_K_M")
        return LlamaCppLLMService(mock_server_pool, mock_model_resolver)

    @pytest.mark.asyncio
    async def test_generate_completion_makes_http_request_and_returns_response(self, service):
        # Arrange
        request = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}
        expected_response = {"choices": [{"message": {"content": "Response"}}]}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.json.return_value = expected_response
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response

            # Act
            result = await service.generate_completion(request)

            # Assert
            mock_client_class.assert_called_once()
            mock_client.post.assert_called_once_with(
                "http://127.0.0.1:8080/v1/chat/completions", json=request, timeout=30.0,
            )
            assert result == expected_response

    @pytest.mark.asyncio
    async def test_generate_completion_with_custom_host_and_port(self, service):
        # Arrange
        # Modify the mock server to have custom host/port
        mock_server = service.server_pool.get_server_for_model.return_value
        mock_server.host = "127.0.0.1"
        mock_server.port = 9090
        request = {"model": "test-model"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = MagicMock()
            mock_client.post.return_value.json.return_value = {}

            # Act
            await service.generate_completion(request)

            # Assert
            mock_client.post.assert_called_once_with(
                "http://127.0.0.1:9090/v1/chat/completions", json=request, timeout=30.0,
            )
