from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from src.frameworks_drivers.config import ServerPoolConfig, Config
from src.shared.errors import GPUAllocationError
from src.shared.health_checker import HealthChecker
from src.shared.logger import Logger
from src.entities.performance_monitor import PerformanceMonitor

# Import GPU-related modules
if TYPE_CHECKING:
    from src.entities.gpu_assignment import GPUAssignment
    from src.frameworks_drivers.gpu_allocator import AdaptiveGPUAllocator
    from src.frameworks_drivers.gpu_detector import GPUDetector
    from src.frameworks_drivers.gpu_monitor import GPUMonitor
    from src.frameworks_drivers.model_repository import ModelRepository

logger = Logger.get(__name__)


class GPUResourceManager:
    """
    Manages GPU resource allocation, reservation, and validation for server pool.
    """

    def __init__(self, gpu_detector, gpu_allocator, model_repository):
        self.gpu_detector = gpu_detector
        self.gpu_allocator = gpu_allocator
        self.model_repository = model_repository
        self.performance_monitor = PerformanceMonitor()
        self._active_gpu_assignments: dict[int, 'GPUAssignment'] = {}  # server_id -> GPUAssignment
        self._reserved_gpu_resources: dict[int, float] = {}  # gpu_id -> reserved_vram_in_gb

    def allocate_gpu_for_model(self, model_identifier: str) -> Optional['GPUAssignment']:
        """
        Allocate GPU resources for a model.

        Args:
            model_identifier: The model identifier
            performance_monitor: Performance monitor to record allocation time

        Returns:
            GPUAssignment if successful, None otherwise
        """
        gpu_assignment = None

        # Start timing GPU allocation
        start_time = time.time()

        try:
            if self.gpu_allocator and self.gpu_detector:
                self.gpu_detector.detect_gpus()

                if not self.gpu_detector.is_gpu_available():
                    logger.info("No GPUs detected, proceeding with CPU-only mode")
                    return None
                else:
                    logger.info(f"Attempting GPU allocation for model: {model_identifier}")

                    # Get model details from repository to estimate VRAM requirement
                    required_vram = self._estimate_model_vram_requirement(model_identifier)

                    if required_vram:
                        logger.info(f"Estimated VRAM requirement for {model_identifier}: {required_vram:.2f}GB")

                        # Get available GPUs for allocation
                        available_gpus = self.gpu_detector.get_available_gpus()
                        logger.debug(f"Available GPUs for allocation: {[f'GPU{gpu.id}({gpu.free_memory:.1f}GB)' for gpu in available_gpus]}")

                        if available_gpus:
                            # Perform GPU allocation using the allocator
                            # Pass model_identifier as gguf_path if it's a local file (GGUF)
                            gguf_path = model_identifier if model_identifier.endswith('.gguf') else None
                            if gguf_path:
                                gpu_assignment = self.gpu_allocator.allocate_gpus(required_vram, available_gpus, gguf_path=gguf_path)
                            else:
                                gpu_assignment = self.gpu_allocator.allocate_gpus(required_vram, available_gpus)
                            if gpu_assignment:
                                logger.debug(f"GPU allocator returned assignment: {gpu_assignment.gpu_ids}, "
                                            f"VRAM required: {gpu_assignment.estimated_vram_required:.2f}GB")
                                # Validate GPU allocation before proceeding
                                if not self._validate_gpu_allocation(gpu_assignment, model_identifier):
                                    logger.warning(f"GPU allocation validation failed for model {model_identifier}, releasing reserved resources")
                                    # Release the reserved resources
                                    self._release_gpu_resources(gpu_assignment.gpu_ids, gpu_assignment.estimated_vram_required)
                                    gpu_assignment = None
                                    logger.info(f"Proceeding with CPU-only for model {model_identifier}")
                                else:
                                    # Reserve GPU resources during allocation process
                                    self._reserve_gpu_resources(gpu_assignment.gpu_ids, gpu_assignment.estimated_vram_required)

                                    logger.info(f"GPU allocated successfully: {gpu_assignment.gpu_ids} for model {model_identifier}, "
                                               f"estimated VRAM: {gpu_assignment.estimated_vram_required:.2f}GB")
                            else:
                                logger.warning(f"GPU allocator could not assign resources for model {model_identifier}")
                                # Check if allocation failed due to insufficient resources
                                available_gpus = self.gpu_detector.get_available_gpus()
                                if available_gpus:
                                    total_available_vram = sum(gpu.free_memory for gpu in available_gpus)
                                    logger.info(f"Total available VRAM: {total_available_vram:.2f}GB, "
                                               f"Required: {required_vram:.2f}GB")
                                    if required_vram > total_available_vram:
                                        error_msg = (f"Insufficient GPU resources for model {model_identifier}. "
                                                   f"Required: {required_vram:.2f}GB, Available: {total_available_vram:.2f}GB")
                                        logger.error(error_msg)
                                        raise GPUAllocationError(error_msg)
                                logger.info(f"Proceeding with CPU-only for model {model_identifier}")
                        else:
                            logger.info("No GPUs available, proceeding with CPU-only")
                    else:
                        logger.warning(f"Could not estimate VRAM for model {model_identifier}, proceeding with CPU-only")
            else:
                logger.info("GPU modules not available, proceeding with CPU-only")
        except GPUAllocationError:
            # Re-raise GPU allocation errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error during GPU allocation for model {model_identifier}: {e}")
            gpu_assignment = None

        # Record allocation time for performance monitoring
        allocation_time_ms = (time.time() - start_time) * 1000
        self.performance_monitor.record_allocation_time(allocation_time_ms, gpu_assignment is not None)
        logger.debug(f"GPU allocation process took {allocation_time_ms:.2f}ms for model {model_identifier}, success: {gpu_assignment is not None}")

        return gpu_assignment

    def register_gpu_assignment(self, server_id: int, gpu_assignment: 'GPUAssignment'):
        """Register a GPU assignment for a server."""
        self._active_gpu_assignments[server_id] = gpu_assignment

    def unregister_gpu_assignment(self, server_id: int):
        """Unregister a GPU assignment for a server."""
        if server_id in self._active_gpu_assignments:
            gpu_assignment = self._active_gpu_assignments[server_id]
            self._release_gpu_resources(gpu_assignment.gpu_ids, gpu_assignment.estimated_vram_required)
            del self._active_gpu_assignments[server_id]

    def _estimate_model_vram_requirement(self, model_identifier: str) -> Optional[float]:
        """
        Estimate VRAM requirement for a model using model repository information.

        Args:
            model_identifier: The model identifier to estimate VRAM for

        Returns:
            Estimated VRAM in GB or None if estimation fails
        """
        if not self.model_repository:
            logger.warning("Model repository not available for VRAM estimation")
            return None

        try:
            # Get model details from repository
            model = self.model_repository.get_model(model_identifier)
            logger.debug(f"Retrieved model details from repository: {model_identifier}, parameters: {model.parameters}, variant: {model.variant}")

            # Use the GPU allocator's VRAM estimation method
            if self.gpu_allocator and hasattr(self.gpu_allocator, 'estimate_model_vram'):
                estimated_vram = self.gpu_allocator.estimate_model_vram(
                    model_parameters=model.parameters,
                    model_variant=model.variant
                )
                if estimated_vram:
                    logger.debug(f"Estimated VRAM for model {model_identifier}: {estimated_vram:.2f}GB")
                else:
                    logger.warning(f"VRAM estimation returned None for model {model_identifier}")
                return estimated_vram
            else:
                logger.warning("GPU allocator not available or does not have estimate_model_vram method")
                return None

        except KeyError as e:
            logger.warning(f"Model {model_identifier} not found in repository: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error estimating VRAM for model {model_identifier}: {e}")
            return None

    def _reserve_gpu_resources(self, gpu_ids: list[int], vram_gb: float) -> None:
        """
        Reserve GPU resources during allocation to prevent race conditions.

        Args:
            gpu_ids: List of GPU IDs to reserve resources for
            vram_gb: Amount of VRAM to reserve in GB
        """
        for gpu_id in gpu_ids:
            if gpu_id in self._reserved_gpu_resources:
                self._reserved_gpu_resources[gpu_id] += vram_gb
            else:
                self._reserved_gpu_resources[gpu_id] = vram_gb

        logger.debug(f"Reserved {vram_gb}GB VRAM on GPUs {gpu_ids}")

    def _release_gpu_resources(self, gpu_ids: list[int], vram_gb: float) -> None:
        """
        Release GPU resources when allocation is complete or failed.

        Args:
            gpu_ids: List of GPU IDs to release resources for
            vram_gb: Amount of VRAM to release in GB
        """
        for gpu_id in gpu_ids:
            if gpu_id in self._reserved_gpu_resources:
                self._reserved_gpu_resources[gpu_id] -= vram_gb
                # Remove the entry if reservation reaches zero
                if self._reserved_gpu_resources[gpu_id] <= 0:
                    del self._reserved_gpu_resources[gpu_id]

        logger.debug(f"Released {vram_gb}GB VRAM on GPUs {gpu_ids}")

    def _validate_gpu_allocation(self, gpu_assignment: 'GPUAssignment', model_identifier: str) -> bool:
        """
        Validate GPU allocation before model loading.

        Args:
            gpu_assignment: The GPU assignment to validate
            model_identifier: The model identifier being loaded

        Returns:
            True if allocation is valid, False otherwise
        """
        if not gpu_assignment or not gpu_assignment.gpu_ids:
            logger.warning(f"No GPU IDs in assignment for model {model_identifier}")
            return False

        # Check if all GPUs in the assignment are still available
        if self.gpu_detector:
            available_gpus = self.gpu_detector.get_available_gpus()
            available_gpu_ids = [gpu.id for gpu in available_gpus]

            for gpu_id in gpu_assignment.gpu_ids:
                if gpu_id not in available_gpu_ids:
                    logger.warning(f"GPU {gpu_id} is no longer available for model {model_identifier}")
                    return False

            # Check if the estimated VRAM is still sufficient considering reserved resources
            for gpu_id in gpu_assignment.gpu_ids:
                gpu_info = next((gpu for gpu in available_gpus if gpu.id == gpu_id), None)
                if gpu_info:
                    reserved_on_gpu = self._reserved_gpu_resources.get(gpu_id, 0.0)
                    available_vram = gpu_info.free_memory - reserved_on_gpu

                    if available_vram < gpu_assignment.estimated_vram_required / len(gpu_assignment.gpu_ids):
                        logger.warning(f"Insufficient VRAM on GPU {gpu_id} for model {model_identifier}. "
                                      f"Required: {gpu_assignment.estimated_vram_required / len(gpu_assignment.gpu_ids):.2f}GB, "
                                      f"Available: {available_vram:.2f}GB")
                        return False

        logger.info(f"GPU allocation validated successfully for model {model_identifier}")
        return True

    def get_reserved_gpu_resources(self) -> dict[int, float]:
        """
        Get the currently reserved GPU resources.

        Returns:
            Dict mapping GPU IDs to reserved VRAM in GB
        """
        return self._reserved_gpu_resources.copy()

    def get_active_gpu_assignments(self) -> dict[int, 'GPUAssignment']:
        """
        Get the current active GPU assignments for monitoring purposes.

        Returns:
            Dict mapping server IDs to their GPU assignments
        """
        return self._active_gpu_assignments.copy()

    def get_performance_stats(self) -> dict[str, float]:
        """
        Get performance statistics for GPU allocation decisions.

        Returns:
            Dict containing performance metrics
        """
        return {
            "average_allocation_time_ms": self.performance_monitor.get_average_allocation_time(),
            "recent_allocation_time_ms": self.performance_monitor.get_recent_allocation_time(),
            "allocation_success_count": self.performance_monitor.allocation_success_count,
            "allocation_failure_count": self.performance_monitor.allocation_failure_count,
            "total_allocations": self.performance_monitor.allocation_success_count + self.performance_monitor.allocation_failure_count
        }


