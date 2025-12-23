import time
import asyncio
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from llama_cpp import Llama
from .config import ServerPoolConfig
from .model_resolver import ModelResolver
from .common_imports import Logger

logger = Logger.get(__name__)


@dataclass
class ServerInstance:
    """Represents a single llama.cpp server instance."""
    id: int
    llama: Optional[Llama] = None
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
            self.servers.append(ServerInstance(id=i))

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
            if server.llama is None and server.is_healthy:
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
        Load a model into a server instance.

        Args:
            server: The server instance to load into.
            model_identifier: The model identifier.

        Returns:
            True if successful, False otherwise.
        """
        logger.debug(f"Loading model {model_identifier} into server {server.id}")
        try:
            repo_id, filename_pattern = ModelResolver.resolve(model_identifier)
            logger.debug(f"Resolved to repo_id={repo_id}, pattern={filename_pattern}")

            # Use Llama.from_pretrained to download and load the model
            # This is already properly wrapped in asyncio.to_thread
            llama = await asyncio.to_thread(
                Llama.from_pretrained,
                repo_id=repo_id,
                filename=filename_pattern,
                # chat_format="openai",  # Set chat format for chat completions
            )

            server.llama = llama
            server.model = model_identifier
            server.is_healthy = True
            logger.info(f"Loaded model {model_identifier} into server {server.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_identifier} into server {server.id}: {e}")
            server.is_healthy = False
            return False

    async def check_health(self) -> None:
        """
        Check the health of all servers.
        Marks unhealthy servers and attempts to recover them.
        """
        for server in self.servers:
            if server.llama is not None:
                try:
                    # Simple health check: try to tokenize a short string
                    # Wrap in asyncio.to_thread since tokenize might be blocking
                    await asyncio.to_thread(server.llama.tokenize, "test".encode('utf-8'))
                    server.is_healthy = True
                except Exception as e:
                    logger.warning(f"Server {server.id} health check failed: {e}")
                    server.is_healthy = False
                    # Attempt to reload the model
                    if server.model:
                        await self._load_model_into_server(server, server.model)

    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get the current status of the server pool.

        Returns:
            Dict with pool information.
        """
        return {
            "total_servers": len(self.servers),
            "healthy_servers": sum(1 for s in self.servers if s.is_healthy),
            "loaded_models": [
                {"server_id": s.id, "model": s.model, "last_used": s.last_used}
                for s in self.servers if s.model is not None
            ],
        }

    def shutdown(self) -> None:
        """Shutdown all servers in the pool."""
        for server in self.servers:
            if server.llama is not None:
                # Llama doesn't have an explicit close method, but we can set to None
                server.llama = None
                server.model = None
        logger.info("Server pool shutdown complete")