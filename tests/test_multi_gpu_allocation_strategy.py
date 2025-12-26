"""
Tests for MultiGPUAllocationStrategy functionality.
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


class TestMultiGPUAllocationStrategy:
    """Test cases for MultiGPUAllocationStrategy."""
    
    def test_multi_gpu_allocation_two_gpus(self):
        """Test multi-GPU allocation across two GPUs."""
        strategy = MultiGPUAllocationStrategy()

        available_gpus = [
            GPU(
                id=0,
                name="GPU 0",
                total_memory=8.0,
                free_memory=4.0,
                used_memory=4.0,
                utilization=50.0,
                assigned_models=[]
            ),
            GPU(
                id=1,
                name="GPU 1",
                total_memory=8.0,
                free_memory=5.0,
                used_memory=3.0,
                utilization=35.0,
                assigned_models=[]
            )
        ]

        # Request 8GB - can be satisfied by both GPUs (4+5=9GB)
        assignment = strategy.allocate_gpus(
            required_vram=8.0,
            available_gpus=available_gpus
        )

        assert assignment is not None
        assert set(assignment.gpu_ids) == {0, 1}  # Both GPUs
        assert assignment.estimated_vram_required == 8.8  # 8.0 * 1.1 headroom
        # Verify tensor_splits computation: proportional to free memory
        assert len(assignment.tensor_splits) == 2
        assert abs(sum(assignment.tensor_splits) - 1.0) < 1e-6  # Should sum to 1.0
        expected_splits = [5.0/9.0, 4.0/9.0]  # 5GB/9GB total, 4GB/9GB total (sorted by free memory desc)
        for actual, expected in zip(assignment.tensor_splits, expected_splits):
            assert abs(actual - expected) < 1e-6
    
    def test_multi_gpu_allocation_three_gpus(self):
        """Test multi-GPU allocation across three GPUs."""
        strategy = MultiGPUAllocationStrategy()

        available_gpus = [
            GPU(id=0, name="GPU 0", total_memory=8.0, free_memory=2.0, used_memory=6.0, utilization=75.0, assigned_models=[]),
            GPU(id=1, name="GPU 1", total_memory=8.0, free_memory=3.0, used_memory=5.0, utilization=60.0, assigned_models=[]),
            GPU(id=2, name="GPU 2", total_memory=8.0, free_memory=4.0, used_memory=4.0, utilization=50.0, assigned_models=[])
        ]

        # Request 6GB - needs GPUs 1 and 2 (3+4=7GB with headroom 6.6)
        assignment = strategy.allocate_gpus(
            required_vram=6.0,
            available_gpus=available_gpus
        )

        assert assignment is not None
        assert set(assignment.gpu_ids) == {1, 2}
        assert assignment.estimated_vram_required == pytest.approx(6.6, abs=1e-10)  # 6.0 * 1.1
        # Verify tensor_splits computation: proportional to free memory of selected GPUs
        assert len(assignment.tensor_splits) == 2
        assert abs(sum(assignment.tensor_splits) - 1.0) < 1e-6  # Should sum to 1.0
        expected_splits = [4.0/7.0, 3.0/7.0]  # 4GB/7GB total, 3GB/7GB total for GPUs 2 and 1 (sorted desc)
        for actual, expected in zip(assignment.tensor_splits, expected_splits):
            assert abs(actual - expected) < 1e-6
    
    def test_multi_gpu_allocation_insufficient_total(self):
        """Test multi-GPU allocation when total VRAM is insufficient."""
        strategy = MultiGPUAllocationStrategy()
        
        available_gpus = [
            GPU(id=0, name="GPU 0", total_memory=8.0, free_memory=3.0, used_memory=5.0, utilization=60.0, assigned_models=[]),
            GPU(id=1, name="GPU 1", total_memory=8.0, free_memory=2.0, used_memory=6.0, utilization=75.0, assigned_models=[])
        ]
        
        # Request 7GB - total available is only 5GB (3+2)
        assignment = strategy.allocate_gpus(
            required_vram=7.0,
            available_gpus=available_gpus
        )
        
        assert assignment is None

    def test_multi_gpu_mixed_memory_capacities(self):
        """Test multi-GPU allocation with GPUs of different memory capacities."""
        strategy = MultiGPUAllocationStrategy()

        available_gpus = [
            GPU(id=0, name="RTX 3060", total_memory=12.0, free_memory=10.0, used_memory=2.0, utilization=16.7, assigned_models=[]),
            GPU(id=1, name="RTX 4080", total_memory=16.0, free_memory=14.0, used_memory=2.0, utilization=12.5, assigned_models=[]),
            GPU(id=2, name="RTX 4090", total_memory=24.0, free_memory=20.0, used_memory=4.0, utilization=16.7, assigned_models=[])
        ]

        # Request 21GB - should select RTX 4080 and RTX 4090 (14+20=34GB), since single GPU max is 20GB
        assignment = strategy.allocate_gpus(
            required_vram=21.0,
            available_gpus=available_gpus
        )

        assert assignment is not None
        assert set(assignment.gpu_ids) == {1, 2}  # RTX 4080 and RTX 4090
        assert assignment.estimated_vram_required == pytest.approx(23.1, abs=1e-10)  # 21.0 * 1.1
        # Verify tensor_splits: proportional to free memory of selected GPUs
        assert len(assignment.tensor_splits) == 2
        assert abs(sum(assignment.tensor_splits) - 1.0) < 1e-6
        expected_splits = [20.0/34.0, 14.0/34.0]  # 20GB/34GB total, 14GB/34GB total (sorted desc)
        for actual, expected in zip(assignment.tensor_splits, expected_splits):
            assert abs(actual - expected) < 1e-6

    def test_multi_gpu_with_gpu_failures(self):
        """Test multi-GPU allocation when some GPUs have very low memory."""
        strategy = MultiGPUAllocationStrategy()

        available_gpus = [
            GPU(id=0, name="GPU 0", total_memory=8.0, free_memory=0.5, used_memory=7.5, utilization=93.8, assigned_models=[]),  # Nearly full
            GPU(id=1, name="GPU 1", total_memory=8.0, free_memory=6.0, used_memory=2.0, utilization=25.0, assigned_models=[]),
            GPU(id=2, name="GPU 2", total_memory=8.0, free_memory=5.0, used_memory=3.0, utilization=37.5, assigned_models=[])
        ]

        # Request 10GB - should select GPUs 1 and 2 (6+5=11GB), skip GPU 0
        assignment = strategy.allocate_gpus(
            required_vram=10.0,
            available_gpus=available_gpus
        )

        assert assignment is not None
        assert set(assignment.gpu_ids) == {1, 2}
        assert assignment.estimated_vram_required == 11.0  # 10.0 * 1.1
        # Verify tensor_splits
        assert len(assignment.tensor_splits) == 2
        assert abs(sum(assignment.tensor_splits) - 1.0) < 1e-6
        expected_splits = [6.0/11.0, 5.0/11.0]
        for actual, expected in zip(assignment.tensor_splits, expected_splits):
            assert abs(actual - expected) < 1e-6

    def test_multi_gpu_insufficient_even_with_all_gpus(self):
        """Test multi-GPU allocation when even all GPUs combined don't have enough memory."""
        strategy = MultiGPUAllocationStrategy()

        available_gpus = [
            GPU(id=0, name="GPU 0", total_memory=8.0, free_memory=3.0, used_memory=5.0, utilization=62.5, assigned_models=[]),
            GPU(id=1, name="GPU 1", total_memory=8.0, free_memory=2.0, used_memory=6.0, utilization=75.0, assigned_models=[]),
            GPU(id=2, name="GPU 2", total_memory=8.0, free_memory=1.0, used_memory=7.0, utilization=87.5, assigned_models=[])
        ]

        # Request 10GB - total available is only 6GB (3+2+1)
        assignment = strategy.allocate_gpus(
            required_vram=10.0,
            available_gpus=available_gpus
        )

        assert assignment is None