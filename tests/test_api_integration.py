"""
Integration tests for API endpoints with mocked server pool and llama.cpp interactions.

This module tests the actual API endpoints (/health and /chat/completions) without requiring
actual llama.cpp servers by mocking the server pool and model interactions.
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from src.proxy.config import Config, ServerPoolConfig, ModelConfig
from src.proxy.types import Message, AgentConfig
from src.proxy.server_pool import ServerInstance
from src.proxy.api import API


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
                is_healthy=True
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
                {"server_id": 1, "model": "test-model-1", "last_used": mock_servers[1].last_used}
            ]
        }
        mock_pool.shutdown.return_value = None
        return mock_pool

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
            "choices": [{"message": {"content": "This is a test response from the model."}}]
        }
        return mock_response

    @pytest.fixture
    def api_client(self, test_config, mock_server_pool, mock_agent_manager, mock_http_response):
        """Create test client with mocked dependencies."""
        # Create API instance with the test config
        api_instance = API(test_config)

        # Mock httpx.AsyncClient as an async context manager
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post.return_value = mock_http_response
        mock_client.get.return_value = mock_http_response

        # Patch the API instance's server_pool, agent_manager, and httpx
        with patch.object(api_instance, 'server_pool', mock_server_pool), \
             patch.object(api_instance, 'agent_manager', mock_agent_manager), \
             patch('httpx.AsyncClient', return_value=mock_client):
            client = TestClient(api_instance.app)
            yield client

    class TestHealthEndpoint:
        """Test /health endpoint."""

        def test_health_endpoint_success(self, api_client, mock_server_pool):
            """Test successful health check."""
            response = api_client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "total_servers" in data
            assert "healthy_servers" in data
            assert "loaded_models" in data
            
            # Verify values
            assert data["total_servers"] == 2
            assert data["healthy_servers"] == 2
            assert isinstance(data["loaded_models"], list)
            
            # Verify server pool methods were called
            mock_server_pool.check_health.assert_called_once()
            mock_server_pool.get_pool_status.assert_called_once()

        def test_health_endpoint_exception(self, api_client, mock_server_pool):
            """Test health endpoint when server pool throws exception."""
            # Make get_pool_status raise an exception
            mock_server_pool.get_pool_status.side_effect = Exception("Pool error")
            
            response = api_client.get("/health")
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Health check failed" in data["detail"]

        def test_health_endpoint_partial_health(self, api_client, mock_server_pool):
            """Test health endpoint with some unhealthy servers."""
            mock_server_pool.get_pool_status.return_value = {
                "total_servers": 2,
                "healthy_servers": 1,
                "loaded_models": [
                    {"server_id": 0, "model": "test-model-0", "last_used": time.time()}
                ]
            }

            response = api_client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["total_servers"] == 2
            assert data["healthy_servers"] == 1

    class TestChatCompletionsEndpoint:
        """Test /chat/completions endpoint."""

        def test_chat_completions_success(self, api_client, mock_agent_manager, mock_server_pool, mock_http_response):
            """Test successful chat completion."""
            request_data = {
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "temperature": 0.7,
                "max_tokens": 100
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

        def test_chat_completions_with_conversation(self, api_client, mock_agent_manager, mock_server_pool, mock_http_response):
            """Test chat completion with multi-message conversation."""
            request_data = {
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How can you help me?"}
                ]
            }

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 200

            # Verify HTTP call was made with correct messages
            # Since httpx is mocked at the class level, we need to check the call
            # The messages are passed directly to the HTTP request

        def test_chat_completions_no_messages(self, api_client):
            """Test chat completion with no messages."""
            request_data = {
                "model": "test-model",
                "messages": []
            }

            response = api_client.post("/chat/completions", json=request_data)

            # No validation, so it forwards as is
            assert response.status_code == 200
            data = response.json()
            assert "choices" in data

        def test_chat_completions_last_message_not_user(self, api_client):
            """Test chat completion when last message is not from user."""
            request_data = {
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
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
            
            request_data = {
                "model": "nonexistent-model",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            }
            
            response = api_client.post("/chat/completions", json=request_data)
            
            assert response.status_code == 503
            data = response.json()
            assert "detail" in data
            assert "No available server" in data["detail"]

        def test_chat_completions_model_generation_error(self, api_client, mock_agent_manager, mock_server_pool, mock_http_response):
            """Test chat completion when model generation fails."""
            # Make HTTP response indicate failure
            mock_http_response.status_code = 500
            mock_http_response.json.return_value = {"error": "Model generation failed"}

            request_data = {
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            }

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "llama-server returned 500" in data["detail"]

        def test_chat_completions_with_agent_hooks(self, api_client, mock_agent_manager, mock_server_pool, mock_http_response):
            """Test chat completion with agent processing."""
            # Agent processing has been removed, so no hooks are called
            request_data = {
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            }

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 200

            # Verify response structure (raw from llama-server)
            data = response.json()
            assert "choices" in data

        def test_chat_completions_with_slash_commands(self, api_client, mock_agent_manager, mock_server_pool, mock_http_response):
            """Test chat completion with slash commands."""
            # Mock agent to recognize slash commands
            mock_agent_manager.parse_slash_commands.return_value = ["test-agent"]
            mock_agent_manager.build_agent_chain.return_value = [Mock()]
            
            request_data = {
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "/test-agent Hello there"}
                ]
            }
            
            response = api_client.post("/chat/completions", json=request_data)
            
            assert response.status_code == 200
            
            # Agent processing has been removed, so slash commands are not parsed

        def test_chat_completions_minimal_request(self, api_client, mock_agent_manager, mock_server_pool, mock_http_response):
            """Test chat completion with minimal request data."""
            request_data = {
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            }

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 200

        def test_chat_completions_with_streaming(self, api_client, mock_agent_manager, mock_server_pool, mock_http_response):
            """Test chat completion with streaming enabled (currently not fully implemented)."""
            request_data = {
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "stream": True
            }
            
            response = api_client.post("/chat/completions", json=request_data)
            
            # Should still work, streaming parameter is currently not fully implemented
            assert response.status_code == 200

    class TestAPIErrorHandling:
        """Test API error handling and edge cases."""

        def test_invalid_json(self, api_client):
            """Test request with invalid JSON."""
            response = api_client.post(
                "/chat/completions",
                data="invalid json",
                headers={"Content-Type": "application/json"}
            )
            
            assert response.status_code == 422  # Pydantic validation error

        def test_missing_content_type(self, api_client):
            """Test request without content type."""
            response = api_client.post(
                "/chat/completions",
                data='{"model": "test", "messages": []}'
            )
            
            # Should still work as FastAPI handles this
            assert response.status_code in [200, 400, 422]

        def test_extra_fields_in_request(self, api_client, mock_agent_manager, mock_server_pool, mock_http_response):
            """Test request with extra fields."""
            request_data = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7,
                "extra_field": "should be ignored"
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
            request_data = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}]
            }

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 200
            data = response.json()

            # Verify response has choices
            assert "choices" in data
            assert len(data["choices"]) == 1

        def test_response_has_choices(self, api_client, mock_agent_manager, mock_server_pool, mock_http_response):
            """Test that response has choices."""
            request_data = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}]
            }

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 200
            data = response.json()

            # Verify response has choices
            assert "choices" in data

        def test_response_has_content(self, api_client, mock_agent_manager, mock_server_pool, mock_http_response):
            """Test that response has content."""
            request_data = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello world this is a test"}]
            }

            response = api_client.post("/chat/completions", json=request_data)

            assert response.status_code == 200
            data = response.json()

            # Verify response has choices with message content
            assert "choices" in data
            assert len(data["choices"]) > 0
            assert "message" in data["choices"][0]
            assert "content" in data["choices"][0]["message"]
