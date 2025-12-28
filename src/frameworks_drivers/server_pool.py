from __future__ import annotations

import asyncio
import subprocess
import time
from typing import Any, Optional

from src.frameworks_drivers.config import ServerPoolConfig, Config
from src.frameworks_drivers.gpu.gpu_resource_manager import GPUResourceManager
from src.frameworks_drivers.server_instance import ServerInstance
from src.frameworks_drivers.server_lifecycle_manager import ServerLifecycleManager
from src.shared.errors import GPUAllocationError
from src.shared.logger import Logger
from src.entities.performance_monitor import PerformanceMonitor

# Import GPU-related modules
from src.frameworks_drivers.model_repository import ModelRepository

logger = Logger.get(__name__)


class ServerPool:
    """
    Manages a fixed-capacity pool of llama.cpp servers with lazy initialization,
    model loading, and health monitoring.
    """

    def __init__(self, config: ServerPoolConfig, model_repository: Optional[ModelRepository] = None, full_config: Optional[Config] = None):
        self.config = config
        self.model_repository = model_repository  # Store model repository reference
        self.lock = asyncio.Lock()  # Thread safety for concurrent access
        # Import GPU-related modules inside the initialization to handle import errors gracefully
        gpu_detector = None
        allocate_gpu_resources_use_case = None
        gpu_monitor = None
        try:
            from src.frameworks_drivers.gpu.gpu_detector import GPUDetector
            from src.use_cases.allocate_gpu_resources import AllocateGPUResources
            from src.frameworks_drivers.gpu.gpu_monitor import GPUMonitor
            gpu_detector = GPUDetector(full_config)
            allocate_gpu_resources_use_case = AllocateGPUResources()
            gpu_monitor = GPUMonitor(full_config)
        except ImportError as e:
            # If GPU modules are not available, set them to None to fall back to CPU-only operation
            logger.warning(f"GPU modules not available, falling back to CPU-only mode: {e}")
        except Exception as e:
            # Handle any other exceptions during GPU module initialization (like pynvml issues)
            logger.warning(f"GPU modules initialization failed, falling back to CPU-only mode: {e}")

        # Performance monitoring - single instance shared with GPU manager
        performance_monitor = PerformanceMonitor()

        # Create GPU resource manager
        self.gpu_manager = GPUResourceManager(gpu_detector, allocate_gpu_resources_use_case, model_repository, performance_monitor) if gpu_detector and allocate_gpu_resources_use_case else None
        self.gpu_monitor = gpu_monitor

        # Create server lifecycle manager
        self.server_manager = ServerLifecycleManager(config, self.gpu_manager)

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
        async with self.lock:
            logger.debug(f"get_server_for_model called with {model_identifier}")
            # First, try to find a server already loaded with this model
            for server in self.server_manager.servers:
                if server.model == model_identifier and server.is_healthy:
                    server.last_used = time.time()
                    return server

            # Then, find an idle server (no model loaded)
            for server in self.server_manager.servers:
                if server.process is None and server.is_healthy:
                    try:
                        if await self.server_manager._load_model_into_server(server, model_identifier):
                            server.last_used = time.time()
                            return server
                    except GPUAllocationError:
                        # If GPU allocation failed, continue to try other servers
                        logger.warning(f"GPU allocation failed for server {server.id}, trying next server")
                        continue

            # Finally, find the least recently used server to evict
            # Sort by last_used, ascending (oldest first)
            available_servers = [s for s in self.server_manager.servers if s.is_healthy]
            if available_servers:
                available_servers.sort(key=lambda s: s.last_used)
                oldest_server = available_servers[0]
                try:
                    if await self.server_manager._load_model_into_server(oldest_server, model_identifier):
                        oldest_server.last_used = time.time()
                        return oldest_server
                except GPUAllocationError:
                    # If GPU allocation failed, return None to indicate failure
                    logger.error(f"GPU allocation failed for all available servers when requesting model {model_identifier}")
                    return None

            return None
            
    async def check_health(self) -> None:
        """
        Check the health of all servers.
        Marks unhealthy servers and attempts to recover them.
        """
        async with self.lock:
            await self.server_manager.check_health()

    def get_pool_status(self) -> dict[str, Any]:
        """
        Get the current status of the server pool with GPU information if available.

        Returns:
            Dict with pool information.
        """
        # Note: This method is synchronous, but since it's called from async context, we assume it's ok.
        # If needed, make it async and use lock.
        result = {
            "total_servers": len(self.server_manager.servers),
            "healthy_servers": sum(1 for s in self.server_manager.servers if s.is_healthy),
            "loaded_models": [
                {
                    "server_id": s.id,
                    "model": s.model,
                    "last_used": s.last_used,
                    "gpu_assignment": s.gpu_assignment.model_dump() if s.gpu_assignment else None
                }
                for s in self.server_manager.servers
                if s.model is not None
            ],
        }
        
        # Add GPU pool status if available
        if self.gpu_monitor:
            try:
                from src.entities.gpu_pool_status import GPUPoolStatus
                gpu_pool_status = GPUPoolStatus(
                    total_gpus=self.gpu_monitor.get_gpu_count(),
                    available_gpus=len([g for g in self.gpu_monitor.get_all_gpus() if g.free_memory > 0.1]),
                    total_memory=sum(g.total_memory for g in self.gpu_monitor.get_all_gpus()),
                    used_memory=sum(g.used_memory for g in self.gpu_monitor.get_all_gpus()),
                    free_memory=sum(g.free_memory for g in self.gpu_monitor.get_all_gpus()),
                    gpus=self.gpu_monitor.get_all_gpus(),
                    utilization_average=sum(g.utilization for g in self.gpu_monitor.get_all_gpus()) / len(self.gpu_monitor.get_all_gpus()) if self.gpu_monitor.get_all_gpus() else 0.0
                )
                result["gpu_pool_status"] = gpu_pool_status.model_dump()
            except Exception as e:
                logger.warning(f"Could not get GPU pool status: {e}")
        
        return result


    def shutdown(self) -> None:
        """Shutdown all servers in the pool."""
        self.server_manager.shutdown()

        # Clean up GPU monitoring resources
        if self.gpu_monitor:
            try:
                self.gpu_monitor.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down GPU monitor: {e}")

        logger.info("Server pool shutdown complete")

    async def _estimate_model_vram_requirement(self, model_identifier: str) -> float:
        """
        Estimate VRAM requirement for a model.
        
        Args:
            model_identifier: The model identifier to estimate VRAM for
            
        Returns:
            Estimated VRAM requirement in GB
        """
        from src.shared.vram_estimator import VramEstimator
        
        # Extract model variant information from identifier
        model_variant = model_identifier.split('/')[-1]  # Get filename part
        
        # Fallback: estimate from model variant name
        return VramEstimator.estimate_vram_from_model_details(
            parameters=None,  # Will use default estimation
            variant=model_variant
        ) or 4.0 # Default to 4GB if estimation fails

    async def _load_model_into_server(self, server, model_identifier, performance_monitor=None):
        """
        Wrapper to access the _load_model_into_server method from server_manager.
        This is needed for testing purposes where tests patch this method on the ServerPool instance.
        """
        return await self.server_manager._load_model_into_server(server, model_identifier)
    
    def _test_load_model_success(self, server, model_identifier):
        """
        Test helper method that simulates successful model loading.
        This bypasses the complex subprocess mocking required for actual server startup.
        """
        server.model = model_identifier
        server.is_healthy = True
        return True