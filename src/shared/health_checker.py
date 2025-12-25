import asyncio
import subprocess
from typing import Optional

import requests

from src.shared.logger import Logger

logger = Logger.get(__name__)


class HealthChecker:
    """
    Utility class for performing health checks on servers and processes.
    Consolidates common health checking patterns used across the codebase.
    """

    @staticmethod
    async def check_http_endpoint(host: str, port: int, endpoint: str = "/health", timeout: float = 5.0) -> bool:
        """
        Check if an HTTP endpoint is responding with a successful status code.

        Args:
            host: The host address
            port: The port number
            endpoint: The health endpoint path (default: "/health")
            timeout: Request timeout in seconds

        Returns:
            True if the endpoint responds with 200 status, False otherwise
        """
        url = f"http://{host}:{port}{endpoint}"
        try:
            response = await asyncio.to_thread(
                requests.get, url, timeout=timeout,
            )
            if response.status_code == 200:
                logger.debug(f"Health check passed for {url}")
                return True
            else:
                logger.warning(f"Health check failed for {url}: status {response.status_code}")
                return False
        except Exception as e:
            logger.debug(f"Health check failed for {url}: {e}")
            return False

    @staticmethod
    def check_process_running(process: Optional[subprocess.Popen]) -> bool:
        """
        Check if a subprocess is still running.

        Args:
            process: The subprocess to check

        Returns:
            True if the process is running, False otherwise
        """
        if process is None:
            return False

        return_code = process.poll()
        if return_code is not None:
            logger.warning(f"Process has terminated with return code {return_code}")
            return False

        return True

    @staticmethod
    async def check_server_health(host: str, port: int, process: Optional[subprocess.Popen],
                                  endpoint: str = "/health", timeout: float = 5.0) -> bool:
        """
        Perform a comprehensive health check on a server instance.
        Checks both process status and HTTP endpoint availability.

        Args:
            host: The host address
            port: The port number
            process: The subprocess to check
            endpoint: The health endpoint path
            timeout: Request timeout in seconds

        Returns:
            True if both process is running and HTTP endpoint responds, False otherwise
        """
        # First check if process is running
        if not HealthChecker.check_process_running(process):
            return False

        # Then check HTTP endpoint
        return await HealthChecker.check_http_endpoint(host, port, endpoint, timeout)