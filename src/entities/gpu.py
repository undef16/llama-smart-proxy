from typing import List, Optional
from pydantic import BaseModel


class GPU(BaseModel):
    """Represents a GPU device with utilization, memory status, and assigned models."""
    id: int  # GPU device ID
    name: str  # GPU name/model
    total_memory: float  # Total memory in GB
    free_memory: float  # Free memory in GB
    used_memory: float  # Used memory in GB
    utilization: float  # GPU utilization percentage (0-100)
    temperature: Optional[float] = None  # Temperature in Celsius
    power_usage: Optional[float] = None  # Power usage in watts
    assigned_models: List[str]  # Models currently assigned to this GPU
    compute_capability: Optional[str] = None  # CUDA compute capability