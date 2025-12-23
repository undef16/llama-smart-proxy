import time
import asyncio
import subprocess
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from .config import ServerPoolConfig
from .model_resolver import ModelResolver
from .common_imports import Logger

logger = Logger.get(__name__)


@dataclass
class ServerInstance:
    """Represents a single llama-server subprocess instance."""
    id: int
    process: Optional[subprocess.Popen] = None
    port: int = 0
    model: Optional[str] = None  # The model identifier loaded
    last_used: float = 0.0
    is_healthy: bool = True


class ServerPool:
    """
    Manages a fixed-capacity pool of llama.cpp servers with lazy initialization,
    model loading, and health monitoring.
    """

    def __init__(self, config: ServerPoolConfig):
        self.config = config
        self.servers: List[ServerInstance] = []
        self._initialize_servers()

    def _initialize_servers(self) -> None:
        """Initialize the server pool with empty server instances."""
        for i in range(self.config.size):
            port = self.config.port_start + i
            self.servers.append(ServerInstance(id=i, port=port))

    async def get_server_for_model(self, model_identifier: str) -> Optional[ServerInstance]:
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
        logger.debug(f"Starting llama-server for model {model_identifier} on port {server.port}")
        try:
            # Start the llama-server subprocess
            process = subprocess.Popen(
                ['llama-server', '-hf', model_identifier, '--port', str(server.port), '--host', '127.0.0.1', '--log-verbosity', '0'],
                # , '--api'
                # stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE
            )
            server.process = process
            server.model = model_identifier

            # Wait for the server to be ready
            import httpx
            for _ in range(30):  # Wait up to 30 seconds
                await asyncio.sleep(1)
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"http://127.0.0.1:{server.port}/health", timeout=1.0)
                        if response.status_code == 200:
                            server.is_healthy = True
                            logger.info(f"Started llama-server for model {model_identifier} on port {server.port}")
                            return True
                except:
                    continue

            logger.error(f"llama-server did not become ready for model {model_identifier} on port {server.port}")
            server.is_healthy = False
            return False

        except Exception as e:
            logger.error(f"Failed to start llama-server for model {model_identifier} on port {server.port}: {e}")
            server.is_healthy = False
            return False

    async def check_health(self) -> None:
        """
        Check the health of all servers.
        Marks unhealthy servers and attempts to recover them.
        """
        import httpx
        for server in self.servers:
            if server.process is not None:
                try:
                    # Check if process is still running
                    if server.process.poll() is not None:
                        raise Exception("Process has terminated")

                    # Ping the health endpoint
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"http://localhost:{server.port}/health", timeout=5.0)
                        if response.status_code == 200:
                            server.is_healthy = True
                        else:
                            raise Exception(f"Health check returned {response.status_code}")
                except Exception as e:
                    logger.warning(f"Server {server.id} health check failed: {e}")
                    server.is_healthy = False
                    # Attempt to restart the server
                    if server.model:
                        await self._load_model_into_server(server, server.model)

    def get_pool_status(self) -> Dict[str, Any]:
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
                for s in self.servers if s.model is not None
            ],
        }
        return result

    def shutdown(self) -> None:
        """Shutdown all servers in the pool."""
        for server in self.servers:
            if server.process is not None:
                server.process.terminate()
                try:
                    server.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.process.kill()
                server.process = None
                server.model = None
        logger.info("Server pool shutdown complete")