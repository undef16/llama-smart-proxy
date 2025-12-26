"""
GPU status update and refresh mechanism.
"""
import asyncio
import logging
from typing import List
from datetime import datetime

from src.entities.gpu import GPU
from src.entities.gpu_pool_status import GPUPoolStatus
from src.frameworks_drivers.gpu_monitor import GPUMonitor


class GPUStatusUpdater:
    """Handles GPU status updates and refresh mechanisms."""
    
    def __init__(self, gpu_monitor: GPUMonitor):
        self.logger = logging.getLogger(__name__)
        self.gpu_monitor = gpu_monitor
        self.last_updated = None
        self.refresh_interval = 5  # seconds
        self._running = False
        self._update_task = None
    
    def get_current_gpu_status(self) -> GPUPoolStatus:
        """Get the current status of all GPUs in the pool."""
        gpus = self.gpu_monitor.get_all_gpus()
        
        if not gpus:
            return GPUPoolStatus(
                total_gpus=0,
                available_gpus=0,
                total_memory=0.0,
                used_memory=0.0,
                free_memory=0.0,
                gpus=[],
                utilization_average=0.0
            )
        
        total_memory = sum(gpu.total_memory for gpu in gpus)
        used_memory = sum(gpu.used_memory for gpu in gpus)
        free_memory = sum(gpu.free_memory for gpu in gpus)
        available_gpus = sum(1 for gpu in gpus if gpu.free_memory > 0.1)  # Consider GPUs with >0.1GB free as available
        utilization_average = sum(gpu.utilization for gpu in gpus) / len(gpus) if gpus else 0.0
        
        return GPUPoolStatus(
            total_gpus=len(gpus),
            available_gpus=available_gpus,
            total_memory=total_memory,
            used_memory=used_memory,
            free_memory=free_memory,
            gpus=gpus,
            utilization_average=utilization_average
        )
    
    async def start_background_refresh(self):
        """Start background refresh of GPU status."""
        if self._running:
            return
        
        self._running = True
        self.logger.info("Starting GPU status background refresh")
        
        while self._running:
            try:
                await self.refresh_gpu_status()
                await asyncio.sleep(self.refresh_interval)
            except Exception as e:
                self.logger.error(f"Error in GPU status refresh loop: {e}")
                await asyncio.sleep(self.refresh_interval)
    
    async def stop_background_refresh(self):
        """Stop background refresh of GPU status."""
        self.logger.info("Stopping GPU status background refresh")
        self._running = False
    
    async def refresh_gpu_status(self):
        """Manually refresh GPU status."""
        try:
            # The GPUMonitor updates status on each call to get_all_gpus
            # So we just need to call it to refresh
            status = self.get_current_gpu_status()
            self.last_updated = datetime.now()
            self.logger.debug(f"GPU status refreshed. Total GPUs: {status.total_gpus}, "
                            f"Available: {status.available_gpus}, "
                            f"Free Memory: {status.free_memory:.2f}GB")
        except Exception as e:
            self.logger.error(f"Error refreshing GPU status: {e}")
    
    def update_gpu_assignments(self, gpus: List[GPU], assignments) -> List[GPU]:
        """Update GPU assignments with current model assignments."""
        updated_gpus = []
        
        for gpu in gpus:
            # Create a copy of the GPU and update assigned models
            assigned_models = []
            for assignment in assignments:
                if gpu.id in assignment.gpu_ids:
                    assigned_models.append(assignment.estimated_vram_required)
            
            updated_gpu = GPU(
                id=gpu.id,
                name=gpu.name,
                total_memory=gpu.total_memory,
                free_memory=gpu.free_memory,
                used_memory=gpu.used_memory,
                utilization=gpu.utilization,
                temperature=gpu.temperature,
                power_usage=gpu.power_usage,
                assigned_models=assigned_models,
                compute_capability=gpu.compute_capability
            )
            updated_gpus.append(updated_gpu)
        
        return updated_gpus