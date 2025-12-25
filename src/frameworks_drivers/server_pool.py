import asyncio
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from src.frameworks_drivers.config import ServerPoolConfig
from src.shared.health_checker import HealthChecker
from src.shared.logger import Logger

logger = Logger.get(__name__)


@dataclass
class ServerInstance:
    """Represents a single llama-server subprocess instance."""

    id: int
    process: subprocess.Popen | None = None
    port: int = 0
    model: str | None = None  # The model identifier loaded
    last_used: float = 0.0
    is_healthy: bool = True


class ServerPool:
    """
    Manages a fixed-capacity pool of llama.cpp servers with lazy initialization,
    model loading, and health monitoring.
    """

    def __init__(self, config: ServerPoolConfig):
        self.config = config
        self.servers: list[ServerInstance] = []
        self._initialize_servers()

    def _initialize_servers(self) -> None:
        """Initialize the server pool with empty server instances."""
        for i in range(self.config.size):
            port = self.config.port_start + i
            self.servers.append(ServerInstance(id=i, port=port))

    @staticmethod
    def _is_cuda_available() -> bool:
        """Check if CUDA is available on the system."""
        try:
            result = subprocess.run(["nvidia-smi"], check=False, capture_output=True)
            logger.info(f"Is cuda avaialble: {result.returncode == 0}")
            return result.returncode == 0
        except FileNotFoundError:
            return False

    @staticmethod
    def _terminate_process(process: subprocess.Popen | None, timeout: int) -> None:
        """Terminate a subprocess with a timeout, killing if necessary."""
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    async def get_server_for_model(self, model_identifier: str) -> ServerInstance | None:
        """
        Get an available server for the given model.
        Prefers servers already loaded with the model, then idle servers.

        Args:
            model_identifier: The model identifier from the request.

        Returns:
            A ServerInstance if available, None if pool is full and no compatible server.
        """
        logger.debug(f"get_server_for_model called with {model_identifier}")
        # First, try to find a server already loaded with this model
        for server in self.servers:
            if server.model == model_identifier and server.is_healthy:
                server.last_used = time.time()
                return server

        # Then, find an idle server (no model loaded)
        for server in self.servers:
            if server.process is None and server.is_healthy:
                if await self._load_model_into_server(server, model_identifier):
                    server.last_used = time.time()
                    return server

        # Finally, find the least recently used server to evict
        # Sort by last_used, ascending (oldest first)
        available_servers = [s for s in self.servers if s.is_healthy]
        if available_servers:
            available_servers.sort(key=lambda s: s.last_used)
            oldest_server = available_servers[0]
            if await self._load_model_into_server(oldest_server, model_identifier):
                oldest_server.last_used = time.time()
                return oldest_server

        return None

    async def _load_model_into_server(self, server: ServerInstance, model_identifier: str) -> bool:
        """
        Start a llama-server subprocess for the model.

        Args:
            server: The server instance to start.
            model_identifier: The model identifier.

        Returns:
            True if successful, False otherwise.
        """
        logger.info(f"_load_model_into_server called for server {server.id} with model {model_identifier}, current process: {server.process is not None}, healthy: {server.is_healthy}")
        try:
            # Terminate any existing process
            if server.process is not None:
                logger.warning(f"Terminating existing process for server {server.id}")
                self._terminate_process(server.process, 5)
                server.process = None
                logger.info(f"Existing process terminated and cleaned up for server {server.id}")

            logger.info(f"Starting llama-server subprocess for model {model_identifier} on port {server.port}")
            # Start the llama-server subprocess
            cmd = [
                "llama-server",
                "-hf",
                model_identifier,
                "--port",
                str(server.port),
                "--log-disable",
                "--host",
                "127.0.0.1",
            ]  # , '--log-verbosity', '0'
            if self._is_cuda_available() and self.config.gpu_layers > 0:
                cmd.extend(["--n-gpu-layers", str(self.config.gpu_layers)])
            process = subprocess.Popen(
                cmd,
                # stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE
            )
            server.process = process
            server.model = model_identifier
            logger.info(f"Started subprocess for model {model_identifier} on port {server.port}")

            # Wait for the server to be ready
            for attempt in range(120):  # Wait up to 120 seconds for model download
                await asyncio.sleep(1)
                if not HealthChecker.check_process_running(process):
                    logger.warning(f"Process exited during wait on attempt {attempt}")
                    break
                try:
                    if await HealthChecker.check_http_endpoint("127.0.0.1", server.port, "/health", 1.0):
                        server.is_healthy = True
                        logger.info(f"Successfully loaded model {model_identifier} on server {server.id}, port {server.port}")
                        return True
                except Exception as e:
                    logger.debug(f"Health check failed on attempt {attempt}: {e}")
                    continue

            # If not ready, check if process is still running and get stderr
            if process.poll() is None:
                # Process is still running, terminate it
                logger.warning(f"Terminating process after failed health checks for server {server.id}")
                self._terminate_process(process, 5)
                logger.info(f"Failed process terminated for server {server.id}")

            # Read stderr
            stderr_output = process.stderr.read().decode("utf-8", errors="ignore") if process.stderr else ""
            logger.error(
                f"llama-server did not become ready for model {model_identifier} on port {server.port}. Stderr: {stderr_output}",
            )
            server.is_healthy = False
            logger.info(f"Failed to load model {model_identifier} on server {server.id}")
            return False

        except Exception as e:
            logger.error(f"Failed to start llama-server for model {model_identifier} on port {server.port}: {e}")
            if server.process and server.process.stderr:
                stderr_output = server.process.stderr.read().decode("utf-8", errors="ignore")
                logger.error(f"llama-server stderr: {stderr_output}")
            server.is_healthy = False
            return False

    async def check_health(self) -> None:
        """
        Check the health of all servers.
        Marks unhealthy servers and attempts to recover them.
        """
        for server in self.servers:
            if server.process is not None:
                try:
                    # Perform comprehensive health check
                    is_healthy = await HealthChecker.check_server_health(
                        "localhost", server.port, server.process, "/health", 5.0
                    )
                    if is_healthy:
                        server.is_healthy = True
                        logger.info(f"Health check passed for server {server.id}")
                    else:
                        raise Exception("Health check failed")
                except Exception as e:
                    logger.warning(f"Server {server.id} health check failed: {e}")
                    server.is_healthy = False
                    # Attempt to restart the server
                    if server.model:
                        logger.info(f"Attempting recovery for server {server.id} with model {server.model}")
                        success = await self._load_model_into_server(server, server.model)
                        server.is_healthy = success
                        if success:
                            logger.info(f"Recovery successful for server {server.id}")
                        else:
                            logger.warning(f"Recovery failed for server {server.id}")

    def get_pool_status(self) -> dict[str, Any]:
        """
        Get the current status of the server pool.

        Returns:
            Dict with pool information.
        """
        result = {
            "total_servers": len(self.servers),
            "healthy_servers": sum(1 for s in self.servers if s.is_healthy),
            "loaded_models": [
                {"server_id": s.id, "model": s.model, "last_used": s.last_used}
                for s in self.servers
                if s.model is not None
            ],
        }
        return result

    def shutdown(self) -> None:
        """Shutdown all servers in the pool."""
        for server in self.servers:
            if server.process is not None:
                self._terminate_process(server.process, 10)
                server.process = None
                server.model = None
        logger.info("Server pool shutdown complete")