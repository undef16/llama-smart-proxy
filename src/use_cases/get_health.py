from typing import Any, Dict, List, Optional, TypedDict

from src.shared.protocols import ModelRepositoryProtocol, ModelDTO, ServerDTO
from src.frameworks_drivers.gpu_monitor import GPUMonitor
from src.entities.gpu_pool_status import GPUPoolStatus
from src.entities.gpu import GPU
from src.frameworks_drivers.config import Config


class GetHealth:
    def __init__(self, model_repository: ModelRepositoryProtocol, gpu_monitor: GPUMonitor, config: Optional[Config] = None):
        self.model_repository = model_repository
        self.gpu_monitor = gpu_monitor
        self.config = config

    def execute(self) -> dict[str, Any]:
        models = self.model_repository.get_all_models()
        servers: List[ServerDTO] = []
        for model in models:
            # Model objects have id attribute
            model_id = model.id
            model_servers = self.model_repository.get_servers_for_model(model_id)
            servers.extend(model_servers)
        
        # Get GPU information if GPU monitoring is available
        gpu_pool_status = None
        allocation_strategy = None
        if self.config and self.config.gpu:
            allocation_strategy = self.config.gpu.effective_allocation_strategy
        else:
            allocation_strategy = 'single-gpu-preferred'
            
        # Check if GPU monitoring is initialized and working
        gpu_monitor_available = self.gpu_monitor.initialized
        gpu_available = False
            
        if gpu_monitor_available:
            # Create a mapping of GPU IDs to assigned models for the GPU monitor
            gpu_assignments: Dict[int, List[str]] = {}
            for server in servers:
                # Handle both Server objects and ServerDTO dictionaries
                if hasattr(server, 'get'):  # Dictionary-like object
                    gpu_assignment = server.get('gpu_assignment')
                    model_id = server.get('model_id')
                else:  # Server object
                    gpu_assignment = getattr(server, 'gpu_assignment', None)
                    model_id = getattr(server, 'model_id', None)
                    
                if gpu_assignment and hasattr(gpu_assignment, 'get') and gpu_assignment.get('gpu_ids'):
                    for gpu_id in gpu_assignment['gpu_ids']:
                        if gpu_id not in gpu_assignments:
                            gpu_assignments[gpu_id] = []
                        if model_id and model_id not in gpu_assignments[gpu_id]:
                            gpu_assignments[gpu_id].append(model_id)
            
            # Refresh GPU status for each health check
            gpus = self.gpu_monitor.get_all_gpus(gpu_assignments)
            
            # Check if any GPUs are detected
            if gpus:
                gpu_available = True
                total_gpus = len(gpus)
                available_gpus = sum(1 for gpu in gpus if gpu.free_memory > 0)
                total_memory = sum(gpu.total_memory for gpu in gpus)
                used_memory = sum(gpu.used_memory for gpu in gpus)
                free_memory = sum(gpu.free_memory for gpu in gpus)
                utilization_average = sum(gpu.utilization for gpu in gpus) / len(gpus) if len(gpus) > 0 else 0
                
                gpu_pool_status_obj = GPUPoolStatus(
                    total_gpus=total_gpus,
                    available_gpus=available_gpus,
                    total_memory=total_memory,
                    used_memory=used_memory,
                    free_memory=free_memory,
                    gpus=gpus,
                    utilization_average=utilization_average
                )
                
                # Add allocation strategy to the GPU pool status
                gpu_pool_status = gpu_pool_status_obj.model_dump()
                gpu_pool_status["allocation_strategy"] = allocation_strategy
                gpu_pool_status["gpu_available"] = True  # GPUs are available and detected
            else:
                # GPUs are initialized but no GPUs detected
                gpu_pool_status = {
                    "total_gpus": 0,
                    "available_gpus": 0,
                    "total_memory": 0.0,
                    "used_memory": 0.0,
                    "free_memory": 0.0,
                    "gpus": [],
                    "utilization_average": 0.0,
                    "allocation_strategy": allocation_strategy,
                    "gpu_available": False  # GPUs are not available
                }
        else:
            # GPU monitoring is not initialized - this could be due to pynvml not being available
            gpu_pool_status = {
                "total_gpus": 0,
                "available_gpus": 0,
                "total_memory": 0.0,
                "used_memory": 0.0,
                "free_memory": 0.0,
                "gpus": [],
                "utilization_average": 0.0,
                "allocation_strategy": allocation_strategy,
                "gpu_available": False,  # GPUs are not available
                "gpu_monitoring_enabled": False # GPU monitoring is not enabled
            }
        
        response = {"servers": servers}
        # Only include GPU pool status if GPU configuration is present
        if gpu_pool_status and self.config and self.config.gpu:
            response["gpu_pool_status"] = gpu_pool_status

        return response
