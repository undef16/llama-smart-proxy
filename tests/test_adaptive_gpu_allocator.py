"""
Tests for AdaptiveGPUAllocator functionality.
"""
import pytest
from unittest.mock import Mock, patch
import time

from src.entities.gpu import GPU
from src.entities.gpu_assignment import GPUAssignment
from src.frameworks_drivers.gpu_allocator import (
    SingleGPUAllocationStrategy,
    MultiGPUAllocationStrategy,
    AdaptiveGPUAllocator
)
from src.frameworks_drivers.server_pool import ServerPool, GPUAllocationError
from src.entities.performance_monitor import PerformanceMonitor
from src.frameworks_drivers.config import ServerPoolConfig


class TestAdaptiveGPUAllocator:
    """Test cases for AdaptiveGPUAllocator."""
    
    def test_adaptive_allocator_prefers_single_gpu(self):
        """Test that adaptive allocator prefers single GPU when possible."""
        allocator = AdaptiveGPUAllocator()
        
        available_gpus = [
            GPU(id=0, name="GPU 0", total_memory=12.0, free_memory=10.0, used_memory=2.0, utilization=20.0, assigned_models=[]),
            GPU(id=1, name="GPU 1", total_memory=8.0, free_memory=6.0, used_memory=2.0, utilization=25.0, assigned_models=[])
        ]
        
        # Request 8GB - fits in GPU 0, so should use single GPU
        assignment = allocator.allocate_gpus(
            required_vram=8.0,
            available_gpus=available_gpus
        )
        
        assert assignment is not None
        assert len(assignment.gpu_ids) == 1  # Single GPU preferred
        assert assignment.gpu_ids[0] == 0  # GPU 0 has enough memory
    
    def test_adaptive_allocator_falls_back_to_multi_gpu(self):
        """Test that adaptive allocator falls back to multi-GPU when single GPU insufficient."""
        allocator = AdaptiveGPUAllocator()
        
        available_gpus = [
            GPU(id=0, name="GPU 0", total_memory=6.0, free_memory=4.0, used_memory=2.0, utilization=30.0, assigned_models=[]),
            GPU(id=1, name="GPU 1", total_memory=6.0, free_memory=5.0, used_memory=1.0, utilization=15.0, assigned_models=[])
        ]
        
        # Request 8GB - doesn't fit in single GPU, but fits across both
        assignment = allocator.allocate_gpus(
            required_vram=8.0,
            available_gpus=available_gpus
        )
        
        assert assignment is not None
        assert len(assignment.gpu_ids) == 2  # Multi-GPU allocation
        assert set(assignment.gpu_ids) == {0, 1}
    
    def test_adaptive_allocator_insufficient_total_fails(self):
        """Test that adaptive allocator returns None when total VRAM is insufficient."""
        allocator = AdaptiveGPUAllocator()
        
        available_gpus = [
            GPU(id=0, name="GPU 0", total_memory=4.0, free_memory=2.0, used_memory=2.0, utilization=50.0, assigned_models=[]),
            GPU(id=1, name="GPU 1", total_memory=4.0, free_memory=1.0, used_memory=3.0, utilization=75.0, assigned_models=[])
        ]
        
        # Request 8GB - total available is only 3GB (2+1)
        assignment = allocator.allocate_gpus(
            required_vram=8.0,
            available_gpus=available_gpus
        )
        
        assert assignment is None
    
    def test_estimate_model_vram(self):
        """Test model VRAM estimation."""
        allocator = AdaptiveGPUAllocator()
        
        # Test with 7B parameters and Q4_K_M quantization
        estimated = allocator.estimate_model_vram(
            model_parameters=7_000,
            model_variant="model.Q4_K_M.gguf"
        )
        
        assert estimated is not None
        assert estimated > 0
        
        # Test with no parameters (should return None)
        estimated_none = allocator.estimate_model_vram(
            model_parameters=None,
            model_variant="model.Q4_K_M.gguf"
        )
         
        assert estimated_none is None