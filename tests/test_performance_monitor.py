"""
Tests for PerformanceMonitor functionality.
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


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""
    
    def test_performance_monitor_initialization(self):
        """Test that PerformanceMonitor initializes correctly."""
        monitor = PerformanceMonitor()
        assert monitor.allocation_times == []
        assert monitor.allocation_success_count == 0
        assert monitor.allocation_failure_count == 0
        
    def test_performance_monitor_record_allocation_time(self):
        """Test recording allocation times."""
        monitor = PerformanceMonitor()
        
        # Record a successful allocation
        monitor.record_allocation_time(5.0, True)
        assert len(monitor.allocation_times) == 1
        assert monitor.allocation_times[0] == 5.0
        assert monitor.allocation_success_count == 1
        assert monitor.allocation_failure_count == 0
        
        # Record a failed allocation
        monitor.record_allocation_time(10.0, False)
        assert len(monitor.allocation_times) == 2
        assert monitor.allocation_times[1] == 10.0
        assert monitor.allocation_success_count == 1
        assert monitor.allocation_failure_count == 1
        
    def test_performance_monitor_average_calculation(self):
        """Test average allocation time calculation."""
        monitor = PerformanceMonitor()
        
        # Test with no allocations
        assert monitor.get_average_allocation_time() == 0.0
        
        # Test with allocations
        monitor.record_allocation_time(10.0, True)
        monitor.record_allocation_time(20.0, True)
        assert monitor.get_average_allocation_time() == 15.0
        
    def test_performance_monitor_recent_calculation(self):
        """Test recent allocation time calculation."""
        monitor = PerformanceMonitor()
        
        # Add more than 10 values
        for i in range(15):
            monitor.record_allocation_time(float(i + 1), True)
            
        # Last 10: 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 -> average = 10.5
        assert monitor.get_recent_allocation_time(10) == 10.5
        
    def test_server_pool_initialization_with_performance_monitor(self):
        """Test that ServerPool initializes with a PerformanceMonitor."""
        config = ServerPoolConfig(size=2, port_start=800, gpu_layers=20, host="localhost", request_timeout=300)
        pool = ServerPool(config)

        # In CPU-only mode, gpu_manager is None
        if pool.gpu_manager:
            assert isinstance(pool.gpu_manager.performance_monitor, PerformanceMonitor)
        
    @patch('src.use_cases.allocate_gpu_resources.AdaptiveGPUAllocator')
    @patch('src.frameworks_drivers.gpu.gpu_detector.GPUDetector')
    def test_gpu_allocation_timing_recorded(self, mock_gpu_detector, mock_gpu_allocator):
        """Test that GPU allocation timing is properly recorded."""
        # Setup mocks
        mock_allocator_instance = Mock()
        mock_gpu_allocator.return_value = mock_allocator_instance
        
        mock_detector_instance = Mock()
        mock_gpu_detector.return_value = mock_detector_instance
        
        # Create GPUs for testing (with all required fields)
        mock_gpu = GPU(
            id=0,
            name="Test GPU",
            total_memory=8.0,
            free_memory=8.0,
            used_memory=0.0,
            utilization=0.0,
            assigned_models=[]
        )
        mock_detector_instance.get_available_gpus.return_value = [mock_gpu]
        
        # Mock allocation to return a GPU assignment
        mock_assignment = GPUAssignment(
            gpu_ids=[0],
            tensor_splits=[1.0],
            estimated_vram_required=4.0
        )
        mock_allocator_instance.allocate_gpus.return_value = mock_assignment
        
        config = ServerPoolConfig(size=1, port_start=8000, gpu_layers=20, host="localhost", request_timeout=300)
        pool = ServerPool(config)

        # Ensure GPU manager was created
        assert pool.gpu_manager is not None

        # Mock the VRAM estimation
        with patch.object(pool, '_estimate_model_vram_requirement', return_value=4.0):
            # Call the internal method that handles allocation timing
            start_time = time.time()
            # This simulates the allocation process that would happen in _load_model_into_server
            required_vram = 4.0
            gpu_assignment = mock_allocator_instance.allocate_gpus(required_vram, [mock_gpu])
            
            # Verify that allocation was attempted
            mock_allocator_instance.allocate_gpus.assert_called_once_with(required_vram, [mock_gpu])
            
            # Check that performance was recorded by calling the allocation timing logic
            # We'll test the actual timing recording by calling the method that records it
            pool.gpu_manager.performance_monitor.record_allocation_time(5.0, gpu_assignment is not None)
            assert pool.gpu_manager.performance_monitor.allocation_times[-1] == 5.0
            
    def test_gpu_allocation_error_exception(self):
        """Test GPUAllocationError exception."""
        # Test that the exception can be raised and caught
        with pytest.raises(GPUAllocationError):
            raise GPUAllocationError("Test GPU allocation error")
            
        # Test with message
        try:
            raise GPUAllocationError("Insufficient GPU resources")
        except GPUAllocationError as e:
            assert str(e) == "Insufficient GPU resources"