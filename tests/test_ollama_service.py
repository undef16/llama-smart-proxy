from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.frameworks_drivers.ollama_service import OllamaLLMService


class TestOllamaLLMService:
    @pytest.fixture
    def service(self):
        return OllamaLLMService(host="localhost", port=11434, timeout=30.0)

    @pytest.mark.asyncio
    async def test_generate_completion_success(self, service):
        request = {"model": "llama2", "messages": [{"role": "user", "content": "Hello"}], "temperature": 0.7}
        expected_response = {"choices": [{"message": {"role": "assistant", "content": "Hello! How can I help you?"}}]}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = expected_response
            mock_response.raise_for_status = Mock()
            mock_response.status_code = 200
            mock_response.text = '{"choices": [{"message": {"role": "assistant", "content": "Hello! How can I help you?"}}]}'
            mock_post.return_value = mock_response

            result = await service.generate_completion(request)

            assert result == expected_response
            # Verify the call was made with correct arguments
            args, kwargs = mock_post.call_args
            assert args[0] == "http://localhost:11434/v1/chat/completions"
            assert kwargs["json"] == request
            # Note: timeout is passed to AsyncClient constructor, not to post method

    @pytest.mark.asyncio
    async def test_generate_completion_http_error(self, service):
        request = {"model": "llama2", "messages": [{"role": "user", "content": "Hello"}]}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = Exception("HTTP Error")
            mock_response.status_code = 500
            mock_response.text = "HTTP Error"
            mock_post.return_value = mock_response

            with pytest.raises(Exception, match="HTTP Error"):
                await service.generate_completion(request)

    def test_init(self):
        service = OllamaLLMService(host="127.0.0.1", port=8080, timeout=60.0)
        assert service.host == "127.0.0.1"
        assert service.port == 8080
        assert service.timeout == 60.0
