from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Optional
from src.entities.gpu_assignment import GPUAssignment

@dataclass
class ServerInstance:
    """Represents a single llama-server subprocess instance."""

    id: int
    process: subprocess.Popen | None = None
    port: int = 0
    model: str | None = None # The model identifier loaded
    last_used: float = 0.0
    is_healthy: bool = True
    gpu_assignment: Optional['GPUAssignment'] = None # GPU assignment information
