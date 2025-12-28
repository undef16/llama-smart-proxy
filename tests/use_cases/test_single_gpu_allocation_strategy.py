"""
Tests for SingleGPUAllocationStrategy functionality.
"""
import pytest
from unittest.mock import Mock, patch
import time

from src.entities.gpu import GPU
from src.entities.gpu_assignment import GPUAssignment
from src.use_cases.allocate_gpu_resources import (
    SingleGPUAllocationStrategy,
    MultiGPUAllocationStrategy,
    AdaptiveGPUAllocator
)
from src.frameworks_drivers.server_pool import ServerPool, GPUAllocationError
from src.entities.performance_monitor import PerformanceMonitor
from src.frameworks_drivers.config import ServerPoolConfig


class TestSingleGPUAllocationStrategy:
    """Test cases for SingleGPUAllocationStrategy."""
    
    def test_single_gpu_allocation_fits(self):
        """Test single GPU allocation when model fits in one GPU."""
        strategy = SingleGPUAllocationStrategy()
        
        available_gpus = [
            GPU(
                id=0,
                name="GPU 0",
                total_memory=8.0,
                free_memory=6.0,  # 6GB free
                used_memory=2.0,
                utilization=25.0,
                assigned_models=[]
            ),
            GPU(
                id=1,
                name="GPU 1", 
                total_memory=8.0,
                free_memory=4.0,  # 4GB free
                used_memory=4.0,
                utilization=50.0,
                assigned_models=[]
            )
        ]
        
        # Request 5GB - should fit in GPU 0
        assignment = strategy.allocate_gpus(
            required_vram=5.0,
            available_gpus=available_gpus
        )
        
        assert assignment is not None
        assert assignment.gpu_ids == [0]
        assert assignment.estimated_vram_required == 5.0
    
    def test_single_gpu_allocation_does_not_fit(self):
        """Test single GPU allocation when model doesn't fit in any GPU."""
        strategy = SingleGPUAllocationStrategy()
        
        available_gpus = [
            GPU(
                id=0,
                name="GPU 0",
                total_memory=8.0,
                free_memory=3.0,  # Only 3GB free
                used_memory=5.0,
                utilization=60.0,
                assigned_models=[]
            )
        ]
        
        # Request 5GB - won't fit in 3GB free GPU
        assignment = strategy.allocate_gpus(
            required_vram=5.0,
            available_gpus=available_gpus
        )
        
        assert assignment is None
    
    def test_single_gpu_allocation_prefer_largest_free(self):
        """Test that single GPU allocation prefers the GPU with most free memory."""
        strategy = SingleGPUAllocationStrategy()
        
        available_gpus = [
            GPU(
                id=0,
                name="GPU 0",
                total_memory=8.0,
                free_memory=4.0,  # 4GB free
                used_memory=4.0,
                utilization=50.0,
                assigned_models=[]
            ),
            GPU(
                id=1,
                name="GPU 1",
                total_memory=12.0,
                free_memory=10.0,  # 10GB free - largest
                used_memory=2.0,
                utilization=15.0,
                assigned_models=[]
            ),
            GPU(
                id=2,
                name="GPU 2",
                total_memory=8.0,
                free_memory=6.0,  # 6GB free
                used_memory=2.0,
                utilization=25.0,
                assigned_models=[]
            )
        ]
        
        # Request 8GB - should choose GPU 1 (10GB free)
        assignment = strategy.allocate_gpus(
            required_vram=8.0,
            available_gpus=available_gpus
        )
        
        assert assignment is not None
        assert assignment.gpu_ids == [1]  # GPU 1 has the most free memory
        assert assignment.estimated_vram_required == 8.0