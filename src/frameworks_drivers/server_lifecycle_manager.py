from __future__ import annotations

import asyncio
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from src.frameworks_drivers.config import ServerPoolConfig, Config
from src.frameworks_drivers.gpu.gpu_resource_manager import GPUResourceManager
from src.frameworks_drivers.server_instance import ServerInstance
from src.shared.errors import GPUAllocationError
from src.shared.health_checker import HealthChecker
from src.shared.logger import Logger
from src.entities.performance_monitor import PerformanceMonitor

# Import GPU-related modules
if TYPE_CHECKING:
    from src.entities.gpu_assignment import GPUAssignment
    from src.use_cases.allocate_gpu_resources import AdaptiveGPUAllocator
    from src.frameworks_drivers.gpu_detector import GPUDetector
    from src.frameworks_drivers.gpu_monitor import GPUMonitor
    from src.frameworks_drivers.model_repository import ModelRepository

logger = Logger.get(__name__)


class ServerLifecycleManager:

    """
    Manages the lifecycle of server instances, including initialization, loading models, and health monitoring.
    """

    def __init__(self, config: ServerPoolConfig, gpu_manager: Optional[GPUResourceManager]):
        self.config = config
        self.gpu_manager = gpu_manager
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
            logger.info(f"Is cuda available: {result.returncode == 0}")
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

    async def _load_model_into_server(self, server: ServerInstance, model_identifier: str) -> bool:
        """
        Start a llama-server subprocess for the model with GPU allocation if available.

        Args:
            server: The server instance to start.
            model_identifier: The model identifier.
            performance_monitor: Performance monitor to record timing.

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
                # Clean up GPU assignment tracking
                if self.gpu_manager:
                    self.gpu_manager.unregister_gpu_assignment(server.id)
                    server.gpu_assignment = None
                logger.info(f"Existing process terminated and cleaned up for server {server.id}")

            # Perform GPU allocation if GPU manager is available
            gpu_assignment = None
            if self.gpu_manager:
                try:
                    gpu_assignment = self.gpu_manager.allocate_gpu_for_model(model_identifier)
                except GPUAllocationError:
                    raise  # Re-raise to be handled by caller

            server.gpu_assignment = gpu_assignment
            if gpu_assignment and self.gpu_manager:
                # Track the GPU assignment
                self.gpu_manager.register_gpu_assignment(server.id, gpu_assignment)

            logger.info(f"Starting llama-server subprocess for model {model_identifier} on port {server.port}")
            # Start the llama-server subprocess
            # Determine if model_identifier is a HuggingFace repo or local file
            cmd, env = self._prepare_cmd_params(server, model_identifier)

            process = subprocess.Popen(
                cmd,
                env=env,
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

            # Read stderr if available
            self._read_std_err(server, model_identifier, process)

            return False

        except GPUAllocationError as e:
            # Specific handling for GPU allocation errors
            logger.error(f"GPU allocation failed for model {model_identifier} on port {server.port}: {e}")
            server.is_healthy = False
            return False

        except Exception as e:
            logger.error(f"Failed to start llama-server for model {model_identifier} on port {server.port}: {e}")
            if server.process and hasattr(server.process, 'stderr') and server.process.stderr:
                try:
                    stderr_output = server.process.stderr.read().decode("utf-8", errors="ignore")
                    logger.error(f"llama-server stderr: {stderr_output}")
                except (AttributeError, TypeError, UnicodeDecodeError):
                    # Handle cases where stderr is not readable or doesn't support the expected interface
                    logger.warning("Could not read stderr from process")
            server.is_healthy = False

            # Clean up GPU assignment if process failed to start properly
            if self.gpu_manager:
                self.gpu_manager.unregister_gpu_assignment(server.id)
                server.gpu_assignment = None

            return False

    def _read_std_err(self, server, model_identifier, process):
        stderr_output = ""
        if hasattr(process, 'stderr') and process.stderr:
            try:
                stderr_output = process.stderr.read().decode("utf-8", errors="ignore")
            except (AttributeError, TypeError, UnicodeDecodeError):
                    # Handle cases where stderr is not readable or doesn't support the expected interface
                stderr_output = ""

        logger.error(
                f"llama-server did not become ready for model {model_identifier} on port {server.port}. Stderr: {stderr_output}",
            )
        server.is_healthy = False
        logger.info(f"Failed to load model {model_identifier} on server {server.id}")

            # Clean up GPU assignment if process failed to start properly
        if self.gpu_manager:
            self.gpu_manager.unregister_gpu_assignment(server.id)
            server.gpu_assignment = None

    def _prepare_cmd_params(self, server, model_identifier):
        if '/' in model_identifier and not model_identifier.endswith('.gguf'):
                # HuggingFace repository format
            cmd = [
                    "llama-server",
                    "-hf",
                    model_identifier,
                    "--port",
                    str(server.port),
                    "--log-disable",
                    "--host",
                    "127.0.0.1",
                ]
        else:
                # Local file format
            cmd = [
                    "llama-server",
                    "-m",
                    model_identifier,
                    "--port",
                    str(server.port),
                    "--log-disable",
                    "--host",
                    "127.0.0.1",
                ]

            # Add GPU parameters if GPU assignment exists
        env = os.environ.copy()
        if server.gpu_assignment and server.gpu_assignment.gpu_ids:
                # Determine n_gpu_layers: use calculated value if available, else config
            n_gpu_layers = server.gpu_assignment.n_gpu_layers if server.gpu_assignment.n_gpu_layers is not None else self.config.gpu_layers
            gpu_ids = server.gpu_assignment.gpu_ids
            tensor_splits = server.gpu_assignment.tensor_splits
            if len(gpu_ids) == 1:
                    # Single GPU: set CUDA_VISIBLE_DEVICES to the single GPU ID, use --split-mode none
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])
                cmd.extend(["--split-mode", "none"])
                cmd.extend(["--n-gpu-layers", str(n_gpu_layers)])
            else:
                    # Multi-GPU: set CUDA_VISIBLE_DEVICES to comma-separated GPU IDs, use --split-mode layer, --tensor-split with ratios
                env['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu_id) for gpu_id in gpu_ids)
                cmd.extend(["--split-mode", "layer"])
                cmd.extend(["--tensor-split", ','.join(str(ratio) for ratio in tensor_splits)])
                cmd.extend(["--n-gpu-layers", str(n_gpu_layers)])
        elif self._is_cuda_available() and self.config.gpu_layers > 0:
                # Fall back to default GPU usage if no specific assignment but CUDA is available
            cmd.extend(["--n-gpu-layers", str(self.config.gpu_layers)])
        return cmd,env

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
                            # Clean up GPU assignment if recovery failed
                            if self.gpu_manager:
                                self.gpu_manager.unregister_gpu_assignment(server.id)
                                server.gpu_assignment = None

    def shutdown(self) -> None:
        """Shutdown all servers in the pool."""
        for server in self.servers:
            if server.process is not None:
                self._terminate_process(server.process, 10)
                server.process = None
                server.model = None
                # Clean up GPU assignment tracking
                if self.gpu_manager:
                    self.gpu_manager.unregister_gpu_assignment(server.id)
                    server.gpu_assignment = None

        logger.info("Server lifecycle manager shutdown complete")
