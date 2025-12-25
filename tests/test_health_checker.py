import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from src.shared.health_checker import HealthChecker


class TestHealthChecker:
    """Test cases for the HealthChecker utility class."""

    @pytest.mark.asyncio
    async def test_check_http_endpoint_success(self):
        """Test successful HTTP endpoint check."""
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_to_thread.return_value = mock_response

            result = await HealthChecker.check_http_endpoint("localhost", 8080, "/health", 1.0)

            assert result is True
            mock_to_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_http_endpoint_bad_status(self):
        """Test HTTP endpoint check with bad status code."""
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_to_thread.return_value = mock_response

            result = await HealthChecker.check_http_endpoint("localhost", 8080, "/health", 1.0)

            assert result is False

    @pytest.mark.asyncio
    async def test_check_http_endpoint_exception(self):
        """Test HTTP endpoint check that raises exception."""
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = Exception("Connection failed")

            result = await HealthChecker.check_http_endpoint("localhost", 8080, "/health", 1.0)

            assert result is False

    def test_check_process_running_none_process(self):
        """Test process check with None process."""
        result = HealthChecker.check_process_running(None)
        assert result is False

    def test_check_process_running_active_process(self):
        """Test process check with active process."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is still running

        result = HealthChecker.check_process_running(mock_process)
        assert result is True

    def test_check_process_running_exited_process(self):
        """Test process check with exited process."""
        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # Process has exited

        result = HealthChecker.check_process_running(mock_process)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_server_health_success(self):
        """Test comprehensive server health check success."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None

        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_to_thread.return_value = mock_response

            result = await HealthChecker.check_server_health("localhost", 8080, mock_process, "/health", 5.0)

            assert result is True

    @pytest.mark.asyncio
    async def test_check_server_health_process_exited(self):
        """Test server health check when process has exited."""
        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # Process exited

        result = await HealthChecker.check_server_health("localhost", 8080, mock_process, "/health", 5.0)

        assert result is False

    @pytest.mark.asyncio
    async def test_check_server_health_http_failed(self):
        """Test server health check when HTTP check fails."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None

        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = Exception("Connection failed")

            result = await HealthChecker.check_server_health("localhost", 8080, mock_process, "/health", 5.0)

            assert result is False

    @pytest.mark.asyncio
    async def test_check_server_health_custom_endpoint(self):
        """Test server health check with custom endpoint."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None

        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_to_thread.return_value = mock_response

            result = await HealthChecker.check_server_health("127.0.0.1", 9000, mock_process, "/custom/health", 2.0)

            assert result is True
            # Verify the correct URL was constructed
            call_args = mock_to_thread.call_args[0]
            assert "http://127.0.0.1:9000/custom/health" in str(call_args)