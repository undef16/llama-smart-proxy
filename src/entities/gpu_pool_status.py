from typing import List
from pydantic import BaseModel

from .gpu import GPU


class GPUPoolStatus(BaseModel):
    """Represents the overall status of the GPU pool for resource tracking."""
    total_gpus: int  # Total number of GPUs in the pool
    available_gpus: int  # Number of GPUs currently available
    total_memory: float  # Total memory across all GPUs in GB
    used_memory: float  # Total used memory across all GPUs in GB
    free_memory: float  # Total free memory across all GPUs in GB
    gpus: List[GPU]  # Detailed information about each GPU
    utilization_average: float  # Average GPU utilization percentage