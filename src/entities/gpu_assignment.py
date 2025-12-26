from typing import List, Optional
from pydantic import BaseModel


class GPUAssignment(BaseModel):
    """Represents GPU assignment for a server instance."""
    gpu_ids: List[int]  # GPU IDs assigned to this server
    tensor_splits: List[float]  # Normalized ratios summing to 1.0 for VRAM distribution across GPUs (llama.cpp --tensor-split)
    estimated_vram_required: float  # Estimated VRAM required in GB
    actual_vram_used: Optional[float] = None  # Actual VRAM used in GB, if available
    n_gpu_layers: Optional[int] = None  # Number of layers to offload to GPU (llama.cpp --n-gpu-layers)