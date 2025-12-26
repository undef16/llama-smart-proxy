# Data Model: GPU Integration for Llama-Smart-Proxy

## Overview
This document defines the data models required for GPU monitoring and allocation in the Llama-Smart-Proxy.

## Enhanced Server Entity

### Server (Extended)
```python
from typing import Literal, List, Optional
from pydantic import BaseModel, ConfigDict, Field

class GPUAssignment(BaseModel):
    """Represents GPU assignment for a server instance."""
    gpu_ids: List[int]  # GPU IDs assigned to this server
    estimated_vram_required: float  # Estimated VRAM required in GB
    actual_vram_used: Optional[float] = None  # Actual VRAM used in GB, if available

class Server(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    host: str
    port: int = Field(ge=1, le=65535)
    model_id: str
    status: Literal["stopped", "running", "error"]
    process: int | None = None
    gpu_assignment: Optional[GPUAssignment] = None  # GPU assignment information
```

## New GPU Entity

### GPU
```python
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
```

## Enhanced Model Entity

### Model (Extended)
```python
import re
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

class Model(BaseModel):
    id: str = Field(min_length=1)
    repo: str
    variant: str | None = None
    backend: Literal["llama.cpp", "ollama"]
    estimated_vram: Optional[float] = None  # Estimated VRAM requirement in GB
    parameters: Optional[int] = None  # Number of parameters (e.g., 7B, 13B)
    quantization: Optional[str] = None  # Quantization level (e.g., Q4_K_M, Q5_K_M)
    
    @field_validator("repo")
    @classmethod
    def validate_repo(cls, v):
        if not re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$", v):
            raise ValueError("repo must be in format user/repo")
        return v
```

## GPU Pool Status

### GPUPoolStatus
```python
from typing import List
from pydantic import BaseModel

class GPUPoolStatus(BaseModel):
    """Represents the overall status of GPU resources in the system."""
    total_gpus: int
    available_gpus: int
    total_vram: float  # Total VRAM in GB
    available_vram: float  # Available VRAM in GB
    gpu_list: List[GPU]  # Detailed information about each GPU
    allocation_strategy: str  # Current allocation strategy (e.g., "single-gpu-preferred")
```

## Validation Rules

1. GPU ID must be non-negative integer
2. Memory values must be positive floats
3. Utilization must be between 0 and 100
4. Server can have zero or more GPU assignments
5. Model VRAM estimate must be positive if specified
6. GPU assignment VRAM requirement must be less than or equal to available GPU memory

## State Transitions

### GPU State Transitions
- Available → Assigned: When model is allocated to GPU
- Assigned → Available: When model is unloaded from GPU
- Available → Error: When GPU becomes unavailable
- Error → Available: When GPU recovers

### Server State Transitions with GPU
- Running (CPU-only) → Running (GPU-assigned): When GPU allocation occurs
- Running (GPU-assigned) → Running (CPU-only): When GPU becomes unavailable