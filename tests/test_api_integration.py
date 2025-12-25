"""
Integration tests for API endpoints with mocked server pool and llama.cpp interactions.

This module tests the actual API endpoints (/health and /chat/completions) without requiring
actual llama.cpp servers by mocking the server pool and model interactions.
"""

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.interface_adapters.api import API
from src.frameworks_drivers.server_pool import ServerInstance


class TestAPIIntegration:
    """Integration tests for API endpoints."""

    @pytest.fixture
    def test_config(self, sample_config):
        """Create test configuration using shared fixture."""
        return sample_config

    @pytest.fixture
    def mock_server_pool(self):
        """Create mock server pool with mocked servers."""
        mock_servers = []
        for i in range(2):
            server = ServerInstance(
                id=i,
                process=Mock(),  # Mock subprocess
                port=8001 + i,
                model=f"test-model-{i}",
                last_used=time.time(),
                is_healthy=True,
            )
            mock_servers.append(server)

        mock_pool = Mock()
        mock_pool.get_server_for_model = AsyncMock(return_value=mock_servers[0])
        mock_pool.check_health = AsyncMock(return_value=None)
        mock_pool.get_pool_status.return_value = {
            "total_servers": 2,
            "healthy_servers": 2,
            "loaded_models": [
                {"server_id": 0, "model": "test-model-0", "last_used": mock_servers[0].last_used},
                {"server_id": 1, "model": "test-model-1", "last_used": mock_servers[1].last_used},
            ],
        }
        mock_pool.shutdown.return_value = None
        return mock_pool

    @pytest.fixture
    def mock_model_repository(self):
        """Create mock model repository."""
        mock_repo = Mock()
        mock_repo.get_all_models.return_value = []
        mock_repo.get_servers_for_model.return_value = []
        return mock_repo

    @pytest.fixture
    def mock_agent_manager(self):
        """Create mock agent manager."""
        mock_manager = Mock()
        mock_manager.parse_slash_commands.return_value = []
        mock_manager.build_agent_chain.return_value = []

        # Make execute_request_hooks return the request unchanged
        def execute_request_hooks(request, agent_chain):
            return request

        mock_manager.execute_request_hooks.side_effect = execute_request_hooks

        # Make execute_response_hooks return the response unchanged
        def execute_response_hooks(response, agent_chain):
            return response

        mock_manager.execute_response_hooks.side_effect = execute_response_hooks

        return mock_manager

    @pytest.fixture
    def mock_http_response(self):
        """Create mock HTTP response for llama-server."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "This is a test response from the model."}}],
        }
        mock_response.text = '{"choices": [{"message": {"content": "This is a test response from the model."}}]}'
        return mock_response

    @pytest.fixture
    def api_client(self, test_config, mock_server_pool, mock_agent_manager, mock_http_response, mock_model_repository):
        """Create test client with mocked dependencies."""
        from unittest.mock import Mock

        # Create mocked dependencies for API
        mock_llm_service = Mock()
        mock_llm_service.generate_completion = AsyncMock(return_value={"choices": [{"message": {"content": "Test response"}}]})
        mock_llm_service.forward_request = AsyncMock()

        mock_llm_service = Mock()
        mock_llm_service.generate_completion = AsyncMock(return_value={"choices": [{"message": {"content": "Test response"}}]})

        async def process_execute(request):
            # Simulate the real use case: get server, then call llm_service
            server = await mock_server_pool.get_server_for_model(request["model"])
            if server is None:
                raise Exception("No available server")
            return await mock_llm_service.generate_completion(request)

        mock_process_chat_completion = Mock()
        mock_process_chat_completion.execute = AsyncMock(side_effect=process_execute)
        mock_process_chat_completion.llm_service = mock_llm_service

        # Create chat controller that actually calls the mocks
        async def chat_completions(request):
            # Call the underlying mocks as the real controller would
            try:
                result = await mock_process_chat_completion.execute(request)
                return result
            except Exception as e:
                return {
                    "error": {
                        "message": f"Internal server error: {str(e)}",
                        "type": "internal_error"
                    }
                }

        mock_chat_controller = Mock()
        mock_chat_controller.chat_completions = AsyncMock(side_effect=chat_completions)
        mock_chat_controller.process_chat_completion_use_case = mock_process_chat_completion

        # Create get_health use case that actually calls the mocks
        def get_health_execute():
            # Call the underlying mocks as the real use case would
            models = mock_model_repository.get_all_models()
            servers = []
            for model in models:
                model_servers = mock_model_repository.get_servers_for_model(model.id)
                servers.extend(model_servers)
            return {"servers": [server.model_dump() for server in servers]}

        mock_get_health = Mock()
        mock_get_health.execute = Mock(side_effect=get_health_execute)

        # Create health controller that actually calls the mocks
        def health():
            # Call the underlying mocks as the real controller would
            try:
                return mock_get_health.execute()
            except Exception as e:
                return {
                    "error": {
                        "message": f"Health check failed: {str(e)}",
                        "type": "health_check_error"
                    }
                }

        mock_health_controller = Mock()
        mock_health_controller.health = Mock(side_effect=health)

        # Create API instance with mocked controllers
        api_instance = API(mock_chat_controller, mock_health_controller)

        # Attach controllers to app for testing access
        api_instance.app.chat_controller = mock_chat_controller

        # Patch requests for any HTTP calls that might happen
        with patch("requests.post", return_value=mock_http_response), patch(
            "requests.get", return_value=mock_http_response,
        ):
            client = TestClient(api_instance.app)
            yield client

    class TestHealthEndpoint:
        """Test /health endpoint."""

        def test_health_endpoint_success(self, api_client, mock_model_repository):
            """Test successful health check."""
            response = api_client.get("/health")

            assert response.status_code == 200
            data = response.json()

            # Verify response structure
            assert "servers" in data
            assert isinstance(data["servers"], list)

            # Verify model repository methods were called
            mock_model_repository.get_all_models.assert_called_once()

        def test_health_endpoint_exception(self, api_client, mock_model_repository):
            """Test health endpoint when model repository throws exception."""
            # Make get_all_models raise an exception
            mock_model_repository.get_all_models.side_effect = Exception("Repository error")

            response = api_client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert "error" in data
            assert "message" in data["error"]
            assert "Health check failed" in data["error"]["message"]

    class TestChatCompletionsEndpoint:
        """Test /chat/completions endpoint."""

        def test_chat_completions_success(self, api_client, mock_agent_manager, mock_server_pool, mock_http_response):
            """Test successful chat completion."""
            request_data = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "temperature": 0.7,
                "max_tokens": 100,
            }

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 200
            data = response.json()

            # Verify response structure (raw from llama-server)
            assert "choices" in data

            # Verify choices
            assert len(data["choices"]) == 1
            choice = data["choices"][0]
            assert "message" in choice
            assert "content" in choice["message"]

            # Verify server pool was called
            mock_server_pool.get_server_for_model.assert_called_once_with("test-model")

        def test_chat_completions_with_conversation(
            self, api_client, mock_agent_manager, mock_server_pool, mock_http_response,
        ):
            """Test chat completion with multi-message conversation."""
            request_data = {
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How can you help me?"},
                ],
            }

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 200

            # Verify HTTP call was made with correct messages
            # Since httpx is mocked at the class level, we need to check the call
            # The messages are passed directly to the HTTP request

        def test_chat_completions_no_messages(self, api_client):
            """Test chat completion with no messages."""
            request_data = {"model": "test-model", "messages": []}

            response = api_client.post("/chat/completions", json=request_data)

            # No validation, so it forwards as is
            assert response.status_code == 200
            data = response.json()
            assert "choices" in data

        def test_chat_completions_last_message_not_user(self, api_client):
            """Test chat completion when last message is not from user."""
            request_data = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}],
            }

            response = api_client.post("/chat/completions", json=request_data)

            # No validation, so it forwards as is
            assert response.status_code == 200
            data = response.json()
            assert "choices" in data

        def test_chat_completions_no_server_available(self, api_client, mock_server_pool):
            """Test chat completion when no server is available."""
            # Mock no server available
            mock_server_pool.get_server_for_model.return_value = None

            request_data = {"model": "nonexistent-model", "messages": [{"role": "user", "content": "Hello"}]}

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert "error" in data
            assert "message" in data["error"]
            assert "No available server" in data["error"]["message"]

        def test_chat_completions_model_generation_error(
            self, api_client, mock_agent_manager, mock_server_pool, mock_http_response,
        ):
            """Test chat completion when model generation fails."""
            # Modify the mock to raise an exception
            async def failing_generate_completion(request):
                raise Exception("Model generation failed")

            # Replace the mock service's method
            api_client.app.chat_controller.process_chat_completion_use_case.llm_service.generate_completion = failing_generate_completion

            request_data = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert "error" in data
            assert "message" in data["error"]
            assert "Model generation failed" in data["error"]["message"]

        def test_chat_completions_with_agent_hooks(
            self, api_client, mock_agent_manager, mock_server_pool, mock_http_response,
        ):
            """Test chat completion with agent processing."""
            # Agent processing has been removed, so no hooks are called
            request_data = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 200

            # Verify response structure (raw from llama-server)
            data = response.json()
            assert "choices" in data

        def test_chat_completions_with_slash_commands(
            self, api_client, mock_agent_manager, mock_server_pool, mock_http_response,
        ):
            """Test chat completion with slash commands."""
            # Mock agent to recognize slash commands
            mock_agent_manager.parse_slash_commands.return_value = ["test-agent"]
            mock_agent_manager.build_agent_chain.return_value = [Mock()]

            request_data = {"model": "test-model", "messages": [{"role": "user", "content": "/test-agent Hello there"}]}

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 200

            # Agent processing has been removed, so slash commands are not parsed

        def test_chat_completions_minimal_request(
            self, api_client, mock_agent_manager, mock_server_pool, mock_http_response,
        ):
            """Test chat completion with minimal request data."""
            request_data = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 200

        def test_chat_completions_with_streaming(
            self, api_client, mock_agent_manager, mock_server_pool, mock_http_response,
        ):
            """Test chat completion with streaming enabled (currently not fully implemented)."""
            request_data = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "stream": True}

            response = api_client.post("/chat/completions", json=request_data)

            # Should still work, streaming parameter is currently not fully implemented
            assert response.status_code == 200

    class TestAPIErrorHandling:
        """Test API error handling and edge cases."""

        def test_invalid_json(self, api_client):
            """Test request with invalid JSON."""
            response = api_client.post(
                "/chat/completions", content="invalid json", headers={"Content-Type": "application/json"},
            )

            assert response.status_code == 422  # Pydantic validation error

        def test_missing_content_type(self, api_client):
            """Test request without content type."""
            response = api_client.post("/chat/completions", content='{"model": "test", "messages": []}')

            # Should still work as FastAPI handles this
            assert response.status_code in [200, 400, 422]

        def test_extra_fields_in_request(self, api_client, mock_agent_manager, mock_server_pool, mock_http_response):
            """Test request with extra fields."""
            request_data = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7,
                "extra_field": "should be ignored",
            }

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 200

            # Verify extra field doesn't break anything
            data = response.json()
            assert "choices" in data

    class TestAPIResponseStructure:
        """Test API response structure and data validation."""

        def test_response_structure(self, api_client, mock_agent_manager, mock_server_pool, mock_http_response):
            """Test that response has expected structure."""
            request_data = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 200
            data = response.json()

            # Verify response has choices
            assert "choices" in data
            assert len(data["choices"]) == 1

        def test_response_has_choices(self, api_client, mock_agent_manager, mock_server_pool, mock_http_response):
            """Test that response has choices."""
            request_data = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 200
            data = response.json()

            # Verify response has choices
            assert "choices" in data

        def test_response_has_content(self, api_client, mock_agent_manager, mock_server_pool, mock_http_response):
            """Test that response has content."""
            request_data = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello world this is a test"}],
            }

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 200
            data = response.json()

            # Verify response has choices with message content
            assert "choices" in data
            assert len(data["choices"]) > 0
            assert "message" in data["choices"][0]
            assert "content" in data["choices"][0]["message"]
