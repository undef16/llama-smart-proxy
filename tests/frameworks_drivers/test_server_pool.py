import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import subprocess
import time

from src.frameworks_drivers.server_pool import ServerPool, ServerInstance, ServerLifecycleManager
from src.frameworks_drivers.config import ServerPoolConfig


class TestServerPool:
    def test_initialization(self, server_pool_config):
        pool = ServerPool(server_pool_config)
        assert len(pool.server_manager.servers) == 2
        assert pool.server_manager.servers[0].id == 0
        assert pool.server_manager.servers[0].port == 8080
        assert pool.server_manager.servers[1].id == 1
        assert pool.server_manager.servers[1].port == 8081
        assert all(server.process is None for server in pool.server_manager.servers)
        assert all(server.model is None for server in pool.server_manager.servers)
        assert all(server.is_healthy for server in pool.server_manager.servers)

    @patch('subprocess.run')
    def test_is_cuda_available_true(self, mock_run):
        mock_run.return_value.returncode = 0
        assert ServerPool._is_cuda_available() is True

    @patch('subprocess.run')
    def test_is_cuda_available_false(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        assert ServerPool._is_cuda_available() is False

    @patch('subprocess.run')
    def test_is_cuda_available_false_returncode(self, mock_run):
        mock_run.return_value.returncode = 1
        assert ServerPool._is_cuda_available() is False

    @pytest.mark.asyncio
    async def test_get_server_for_model_already_loaded(self, server_pool):
        server_pool.server_manager.servers[0].model = 'test-model'
        server_pool.server_manager.servers[0].is_healthy = True
        server = await server_pool.get_server_for_model('test-model')
        assert server == server_pool.server_manager.servers[0]
        assert server.last_used > 0

    @pytest.mark.asyncio
    @patch.object(ServerLifecycleManager, '_load_model_into_server', new_callable=AsyncMock)
    async def test_get_server_for_model_idle_server(self, mock_load, server_pool):
        mock_load.return_value = True
        server = await server_pool.get_server_for_model('test-model')
        assert server == server_pool.server_manager.servers[0]
        mock_load.assert_called_once_with(server_pool.server_manager.servers[0], 'test-model')

    @pytest.mark.asyncio
    @patch.object(ServerLifecycleManager, '_load_model_into_server', new_callable=AsyncMock)
    async def test_get_server_for_model_evict_oldest(self, mock_load, server_pool):
        # Set up servers with models and last_used times
        server_pool.server_manager.servers[0].model = 'model1'
        server_pool.server_manager.servers[0].last_used = time.time() - 100
        server_pool.server_manager.servers[1].model = 'model2'
        server_pool.server_manager.servers[1].last_used = time.time() - 50
        mock_load.return_value = True
        server = await server_pool.get_server_for_model('new-model')
        assert server == server_pool.server_manager.servers[0]  # oldest
        mock_load.assert_called_once_with(server_pool.server_manager.servers[0], 'new-model')

    @pytest.mark.asyncio
    async def test_get_server_for_model_no_available(self, server_pool):
        # All servers unhealthy
        for server in server_pool.server_manager.servers:
            server.is_healthy = False
        server = await server_pool.get_server_for_model('test-model')
        assert server is None

    @pytest.mark.asyncio
    @patch('src.frameworks_drivers.server_pool.ServerLifecycleManager._is_cuda_available', return_value=False)
    @patch('subprocess.Popen')
    @patch('asyncio.to_thread', new_callable=AsyncMock)
    @patch('asyncio.sleep')
    async def test_load_model_into_server_success_no_cuda(self, mock_sleep, mock_to_thread, mock_popen, mock_cuda, server_pool):
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        mock_process.poll.return_value = None
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_to_thread.return_value = mock_response

        server = server_pool.server_manager.servers[0]
        result = await server_pool.server_manager._load_model_into_server(server, 'test-model')
        assert result is True
        assert server.process == mock_process
        assert server.model == 'test-model'
        assert server.is_healthy is True
        # Check cmd doesn't have gpu layers
        call_args = mock_popen.call_args[0][0]
        assert '--n-gpu-layers' not in call_args

    @pytest.mark.asyncio
    @patch('src.frameworks_drivers.server_pool.ServerLifecycleManager._is_cuda_available', return_value=True)
    @patch('subprocess.Popen')
    @patch('asyncio.to_thread', new_callable=AsyncMock)
    @patch('asyncio.sleep')
    async def test_load_model_into_server_success_with_cuda(self, mock_sleep, mock_to_thread, mock_popen, mock_cuda, server_pool):
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        mock_process.poll.return_value = None
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_to_thread.return_value = mock_response

        server = server_pool.server_manager.servers[0]
        result = await server_pool.server_manager._load_model_into_server(server, 'test-model')
        assert result is True
        assert server.process == mock_process
        assert server.model == 'test-model'
        assert server.is_healthy is True
        # Check cmd has gpu layers
        call_args = mock_popen.call_args[0][0]
        assert '--n-gpu-layers' in call_args
        assert '10' in call_args

    @pytest.mark.asyncio
    @patch('src.frameworks_drivers.server_pool.ServerLifecycleManager._is_cuda_available', return_value=False)
    @patch('subprocess.Popen')
    @patch('asyncio.to_thread', new_callable=AsyncMock)
    @patch('asyncio.sleep')
    async def test_load_model_into_server_process_exits_early(self, mock_sleep, mock_to_thread, mock_popen, mock_cuda, server_pool):
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        mock_process.poll.side_effect = [None, None, 1]  # Exits after some loops
        mock_to_thread.side_effect = Exception('Connection failed')

        server = server_pool.server_manager.servers[0]
        result = await server_pool.server_manager._load_model_into_server(server, 'test-model')
        assert result is False
        assert server.is_healthy is False
        # Process exited early, so terminate should not be called
        mock_process.terminate.assert_not_called()

    @pytest.mark.asyncio
    @patch('src.frameworks_drivers.server_pool.ServerLifecycleManager._is_cuda_available', return_value=False)
    @patch('subprocess.Popen')
    @patch('asyncio.to_thread', new_callable=AsyncMock)
    @patch('asyncio.sleep')
    async def test_load_model_into_server_health_check_fails(self, mock_sleep, mock_to_thread, mock_popen, mock_cuda, server_pool):
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        mock_process.poll.return_value = None
        mock_to_thread.side_effect = Exception('Connection failed')

        server = server_pool.server_manager.servers[0]
        result = await server_pool.server_manager._load_model_into_server(server, 'test-model')
        assert result is False
        assert server.is_healthy is False
        mock_process.terminate.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.frameworks_drivers.server_pool.ServerLifecycleManager._is_cuda_available', return_value=False)
    @patch('subprocess.Popen')
    @patch('asyncio.sleep')
    async def test_load_model_into_server_popen_exception(self, mock_sleep, mock_popen, mock_cuda, server_pool):
        mock_popen.side_effect = Exception('Popen failed')

        server = server_pool.server_manager.servers[0]
        result = await server_pool.server_manager._load_model_into_server(server, 'test-model')
        assert result is False
        assert server.is_healthy is False

    @pytest.mark.asyncio
    @patch('src.frameworks_drivers.server_pool.ServerLifecycleManager._is_cuda_available', return_value=False)
    @patch('subprocess.Popen')
    @patch('asyncio.to_thread', new_callable=AsyncMock)
    @patch('asyncio.sleep')
    async def test_load_model_into_server_timeout(self, mock_sleep, mock_to_thread, mock_popen, mock_cuda, server_pool):
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        mock_process.poll.return_value = None
        # Make sleep run 120 times without success
        mock_to_thread.side_effect = Exception('Timeout')

        server = server_pool.server_manager.servers[0]
        result = await server_pool.server_manager._load_model_into_server(server, 'test-model')
        assert result is False
        assert server.is_healthy is False
        mock_process.terminate.assert_called_once()

    @pytest.mark.asyncio
    @patch('asyncio.to_thread', new_callable=AsyncMock)
    async def test_check_health_healthy(self, mock_to_thread, server_pool):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_to_thread.return_value = mock_response

        server_pool.server_manager.servers[0].process = MagicMock()
        server_pool.server_manager.servers[0].process.poll.return_value = None

        await server_pool.check_health()
        assert server_pool.server_manager.servers[0].is_healthy is True

    @pytest.mark.asyncio
    @patch('asyncio.to_thread', new_callable=AsyncMock)
    async def test_check_health_process_exited(self, mock_to_thread, server_pool):
        server_pool.server_manager.servers[0].process = MagicMock()
        server_pool.server_manager.servers[0].process.poll.return_value = 1  # Exited

        await server_pool.check_health()
        assert server_pool.server_manager.servers[0].is_healthy is False

    @pytest.mark.asyncio
    @patch('asyncio.to_thread', new_callable=AsyncMock)
    async def test_check_health_bad_status(self, mock_to_thread, server_pool):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_to_thread.return_value = mock_response

        server_pool.server_manager.servers[0].process = MagicMock()
        server_pool.server_manager.servers[0].process.poll.return_value = None

        await server_pool.check_health()
        assert server_pool.server_manager.servers[0].is_healthy is False

    @pytest.mark.asyncio
    @patch('asyncio.to_thread', new_callable=AsyncMock)
    async def test_check_health_request_exception(self, mock_to_thread, server_pool):
        mock_to_thread.side_effect = Exception('Request failed')

        server_pool.server_manager.servers[0].process = MagicMock()
        server_pool.server_manager.servers[0].process.poll.return_value = None

        await server_pool.check_health()
        assert server_pool.server_manager.servers[0].is_healthy is False

    @pytest.mark.asyncio
    @patch.object(ServerLifecycleManager, '_load_model_into_server', new_callable=AsyncMock)
    async def test_check_health_recovery(self, mock_load, server_pool):
        mock_load.return_value = True
        server_pool.server_manager.servers[0].process = MagicMock()
        server_pool.server_manager.servers[0].process.poll.return_value = 1
        server_pool.server_manager.servers[0].model = 'test-model'

        await server_pool.check_health()
        assert server_pool.server_manager.servers[0].is_healthy is True
        mock_load.assert_called_once_with(server_pool.server_manager.servers[0], 'test-model')

    def test_get_pool_status(self, server_pool):
        server_pool.server_manager.servers[0].model = 'model1'
        server_pool.server_manager.servers[0].last_used = 123.45
        server_pool.server_manager.servers[1].is_healthy = False

        status = server_pool.get_pool_status()
        assert status['total_servers'] == 2
        assert status['healthy_servers'] == 1
        assert len(status['loaded_models']) == 1
        assert status['loaded_models'][0]['server_id'] == 0
        assert status['loaded_models'][0]['model'] == 'model1'

    def test_shutdown(self, server_pool):
        mock_process = MagicMock()
        server_pool.server_manager.servers[0].process = mock_process
        server_pool.server_manager.servers[0].model = 'model1'

        server_pool.shutdown()
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=10)
        assert server_pool.server_manager.servers[0].process is None
        assert server_pool.server_manager.servers[0].model is None