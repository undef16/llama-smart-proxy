import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import subprocess
import time

from src.frameworks_drivers.server_pool import ServerPool, ServerInstance
from src.frameworks_drivers.config import ServerPoolConfig


@pytest.fixture
def server_pool_config():
    return ServerPoolConfig(size=2, host="localhost", port_start=8080, gpu_layers=10, request_timeout=300)


@pytest.fixture
def server_pool(server_pool_config):
    return ServerPool(server_pool_config)


@pytest.fixture
def server_pool_size1():
    config = ServerPoolConfig(size=1, host="localhost", port_start=8080, gpu_layers=10, request_timeout=300)
    return ServerPool(config)


class TestServerPoolIntegration:
    """
    Integration tests for server pool health monitoring, model loading failures,
    and interactions between server pool and health checks.
    """

    @pytest.mark.asyncio
    @patch('asyncio.to_thread', new_callable=AsyncMock)
    @patch.object(ServerPool, '_load_model_into_server', new_callable=AsyncMock)
    async def test_health_monitoring_failure_triggers_recovery(self, mock_load, mock_to_thread, server_pool):
        """Test that health check failure leads to server recovery attempt."""
        # Setup: server with process and model
        server_pool.servers[0].process = MagicMock()
        server_pool.servers[0].process.poll.return_value = None
        server_pool.servers[0].model = 'test-model'
        server_pool.servers[0].is_healthy = True

        # Mock health check failure
        mock_to_thread.side_effect = Exception('Health check failed')

        # Mock successful recovery
        mock_load.return_value = True

        # Call check_health
        await server_pool.check_health()

        # Verify recovery attempt
        mock_load.assert_called_once_with(server_pool.servers[0], 'test-model')

        # After recovery, server should be healthy again
        assert server_pool.servers[0].is_healthy is True

    @pytest.mark.asyncio
    @patch.object(ServerPool, '_load_model_into_server', new_callable=AsyncMock)
    async def test_model_loading_failure_prevents_server_selection(self, mock_load, server_pool):
        """Test that model loading failure prevents unhealthy server from being selected."""
        # Mock loading failure
        mock_load.return_value = False

        # Try to get server for model
        server = await server_pool.get_server_for_model('test-model')

        # Should return None since loading failed
        assert server is None

        # Verify loading was attempted on idle servers
        assert mock_load.call_count == 3
        mock_load.assert_any_call(server_pool.servers[0], 'test-model')
        mock_load.assert_any_call(server_pool.servers[1], 'test-model')

    @pytest.mark.asyncio
    @patch('asyncio.to_thread', new_callable=AsyncMock)
    @patch.object(ServerPool, '_load_model_into_server', new_callable=AsyncMock)
    async def test_server_pool_health_check_integration_recovery_flow(self, mock_load, mock_to_thread, server_pool):
        """Test integration: unhealthy server becomes available after successful recovery."""
        # Setup: server with model, initially healthy
        server_pool.servers[0].process = MagicMock()
        server_pool.servers[0].process.poll.return_value = None
        server_pool.servers[0].model = 'test-model'
        server_pool.servers[0].is_healthy = True

        # First, make health check fail
        mock_to_thread.side_effect = Exception('Health check failed')
        mock_load.return_value = True  # Recovery succeeds

        # Call check_health - should trigger recovery
        await server_pool.check_health()

        # Server should be unhealthy initially, then recovered
        assert server_pool.servers[0].is_healthy is True  # After recovery

        # Now, getting server for same model should return this server
        server = await server_pool.get_server_for_model('test-model')
        assert server == server_pool.servers[0]

    @pytest.mark.asyncio
    @patch('asyncio.to_thread', new_callable=AsyncMock)
    @patch.object(ServerPool, '_load_model_into_server', new_callable=AsyncMock)
    async def test_multiple_health_failures_and_partial_recovery(self, mock_load, mock_to_thread, server_pool):
        """Test multiple servers with mixed health statuses and recovery attempts."""
        # Setup two servers
        for i, server in enumerate(server_pool.servers):
            server.process = MagicMock()
            server.process.poll.return_value = None
            server.model = f'model-{i}'
            server.is_healthy = True

        # Mock health checks: first fails, second succeeds
        mock_to_thread.side_effect = [
            Exception('Server 0 failed'),  # First call for server 0
            MagicMock(status_code=200),    # Second call for server 1
        ]

        # Mock recovery: succeeds for first, fails for second (but second doesn't need recovery)
        mock_load.return_value = True

        # Call check_health
        await server_pool.check_health()

        # Server 0: failed health, recovery attempted and succeeded
        assert server_pool.servers[0].is_healthy is True
        mock_load.assert_called_once_with(server_pool.servers[0], 'model-0')

        # Server 1: healthy, no recovery
        assert server_pool.servers[1].is_healthy is True

    @pytest.mark.asyncio
    @patch.object(ServerPool, '_load_model_into_server', new_callable=AsyncMock)
    async def test_unhealthy_servers_skipped_in_selection(self, mock_load, server_pool):
        """Test that unhealthy servers are skipped during server selection."""
        # Make first server unhealthy
        server_pool.servers[0].is_healthy = False
        server_pool.servers[0].model = 'model1'

        # Mock loading success for second server
        mock_load.return_value = True

        # Get server for different model
        server = await server_pool.get_server_for_model('model2')

        # Should select the healthy idle server (index 1), not the unhealthy one
        assert server == server_pool.servers[1]
        mock_load.assert_called_once_with(server_pool.servers[1], 'model2')

    @pytest.mark.asyncio
    @patch('asyncio.to_thread', new_callable=AsyncMock)
    @patch.object(ServerPool, '_load_model_into_server', new_callable=AsyncMock)
    async def test_recovery_failure_leaves_server_unhealthy(self, mock_load, mock_to_thread, server_pool):
        """Test that failed recovery attempt leaves server unhealthy."""
        # Setup unhealthy server with model
        server_pool.servers[0].process = MagicMock()
        server_pool.servers[0].process.poll.return_value = None
        server_pool.servers[0].model = 'test-model'
        server_pool.servers[0].is_healthy = True

        # Mock health check failure
        mock_to_thread.side_effect = Exception('Health check failed')

        # Mock recovery failure
        mock_load.return_value = False

        # Call check_health
        await server_pool.check_health()

        # Server should remain unhealthy after failed recovery
        assert server_pool.servers[0].is_healthy is False

        # Recovery was attempted
        mock_load.assert_called_once_with(server_pool.servers[0], 'test-model')