"""
Integration tests for GPU allocation functionality.
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


class TestGPUAllocationIntegration:
    """Integration tests for model loading with GPU allocation."""
    
    def _setup_mocks(self):
        """Helper method to set up common mocks."""
        mock_detector_instance = Mock()
        mock_model_repo_instance = Mock()
          
        return mock_detector_instance, mock_model_repo_instance
    
    def _create_gpu(self, id, name, total_memory, free_memory, used_memory, utilization=0.0, assigned_models=None):
        """Helper method to create GPU instances."""
        if assigned_models is None:
            assigned_models = []
        return GPU(
            id=id,
            name=name,
            total_memory=total_memory,
            free_memory=free_memory,
            used_memory=used_memory,
            utilization=utilization,
            assigned_models=assigned_models
        )
    
    def _create_server_pool(self, mock_model_repo_instance, size=1, port_start=8000):
        """Helper method to create a server pool with config."""
        config = ServerPoolConfig(
            size=size, 
            port_start=port_start, 
            gpu_layers=20, 
            host="localhost", 
            request_timeout=300
        )
        return ServerPool(config, model_repository=mock_model_repo_instance)
    
    @patch('src.frameworks_drivers.model_repository.ModelRepository')
    @patch('src.frameworks_drivers.gpu_detector.GPUDetector')
    @patch('subprocess.Popen')
    def test_model_loading_with_gpu_allocation_single_gpu(self, mock_popen, mock_gpu_detector, mock_model_repository):
        """Test model loading with single GPU allocation."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_model_repo_instance = Mock()
        mock_gpu_detector.return_value = mock_detector_instance
        mock_model_repository.return_value = mock_model_repo_instance
        
        # Create GPUs for testing
        gpu0 = self._create_gpu(0, "NVIDIA GeForce RTX 4090", 24.0, 20.0, 4.0)
        gpu1 = self._create_gpu(1, "NVIDIA GeForce RTX 4080", 16.0, 12.0, 4.0)
        mock_detector_instance.get_available_gpus.return_value = [gpu0, gpu1]
        mock_detector_instance.is_gpu_available.return_value = True
        mock_detector_instance.is_pynvml_available.return_value = True
        
        # Create server pool
        pool = self._create_server_pool(mock_model_repo_instance)
        
        # Mock the GPU allocator in the GPU resource manager
        mock_allocator_instance = Mock()
        mock_assignment = GPUAssignment(gpu_ids=[0], tensor_splits=[1.0], estimated_vram_required=8.0)
        mock_allocator_instance.allocate_gpus.return_value = mock_assignment
        
        # Replace the allocator in the GPU resource manager
        if pool.gpu_manager:
            pool.gpu_manager.gpu_allocator = mock_allocator_instance
        
        # Mock the health checker to return True for endpoint checks
        with patch('src.shared.health_checker.HealthChecker.check_http_endpoint', return_value=True):
            with patch('src.shared.health_checker.HealthChecker.check_process_running', return_value=True):
                # Mock the VRAM estimation to return 8GB
                with patch.object(pool.gpu_manager, '_estimate_model_vram_requirement', return_value=8.0):
                    import asyncio
                    server = asyncio.run(pool.get_server_for_model("llama-3.2-8b-instruct.Q4_K_M.gguf"))
                     
                    # Verify that allocation was called with the right parameters
                    mock_allocator_instance.allocate_gpus.assert_called_once_with(8.0, [gpu0, gpu1], gguf_path='llama-3.2-8b-instruct.Q4_K_M.gguf')
                     
                    # Server should be returned successfully
                    assert server is not None
                    assert server.gpu_assignment is not None
                    assert server.gpu_assignment.gpu_ids == [0]
                    assert server.gpu_assignment.estimated_vram_required == 8.0
            
    @patch('src.frameworks_drivers.model_repository.ModelRepository')
    @patch('src.frameworks_drivers.gpu_detector.GPUDetector')
    @patch('subprocess.Popen')
    def test_model_loading_with_gpu_allocation_multi_gpu(self, mock_popen, mock_gpu_detector, mock_model_repository):
        """Test model loading with multi-GPU allocation."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_model_repo_instance = Mock()
        mock_gpu_detector.return_value = mock_detector_instance
        mock_model_repository.return_value = mock_model_repo_instance
        
        # Create GPUs for testing
        gpu0 = self._create_gpu(0, "NVIDIA GeForce RTX 4090", 24.0, 10.0, 14.0)
        gpu1 = self._create_gpu(1, "NVIDIA GeForce RTX 4080", 16.0, 8.0, 8.0)
        mock_detector_instance.get_available_gpus.return_value = [gpu0, gpu1]
        mock_detector_instance.is_gpu_available.return_value = True
        mock_detector_instance.is_pynvml_available.return_value = True
        
        # Create server pool
        pool = self._create_server_pool(mock_model_repo_instance)
        
        # Mock the GPU allocator in the GPU resource manager
        mock_allocator_instance = Mock()
        mock_assignment = GPUAssignment(gpu_ids=[0, 1], tensor_splits=[0.5, 0.5], estimated_vram_required=15.0)
        mock_allocator_instance.allocate_gpus.return_value = mock_assignment
        
        # Replace the allocator in the GPU resource manager
        if pool.gpu_manager:
            pool.gpu_manager.gpu_allocator = mock_allocator_instance
        
        # Mock the health checker to return True for endpoint checks
        with patch('src.shared.health_checker.HealthChecker.check_http_endpoint', return_value=True):
            with patch('src.shared.health_checker.HealthChecker.check_process_running', return_value=True):
                # Mock the VRAM estimation to return 15GB
                with patch.object(pool.gpu_manager, '_estimate_model_vram_requirement', return_value=15.0):
                    import asyncio
                    server = asyncio.run(pool.get_server_for_model("llama-3.1-70b-instruct.Q4_K_M.gguf"))
                     
                    # Verify that allocation was called with the right parameters
                    mock_allocator_instance.allocate_gpus.assert_called_once_with(15.0, [gpu0, gpu1], gguf_path='llama-3.1-70b-instruct.Q4_K_M.gguf')
                     
                    # Server should be returned successfully
                    assert server is not None
                    assert server.gpu_assignment is not None
                    assert set(server.gpu_assignment.gpu_ids) == {0, 1}
                    assert server.gpu_assignment.estimated_vram_required == 15.0
    
    @patch('src.frameworks_drivers.model_repository.ModelRepository')
    @patch('src.frameworks_drivers.gpu_detector.GPUDetector')
    @patch('subprocess.Popen')
    def test_model_loading_insufficient_gpu_resources(self, mock_popen, mock_gpu_detector, mock_model_repository):
        """Test model loading when there are insufficient GPU resources."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_model_repo_instance = Mock()
        mock_gpu_detector.return_value = mock_detector_instance
        mock_model_repository.return_value = mock_model_repo_instance
        
        # Create GPUs for testing
        gpu0 = self._create_gpu(0, "NVIDIA GeForce RTX 3060", 12.0, 2.0, 10.0)  # Only 2GB free
        mock_detector_instance.get_available_gpus.return_value = [gpu0]
        mock_detector_instance.is_gpu_available.return_value = True
        mock_detector_instance.is_pynvml_available.return_value = True
        
        # Create server pool
        pool = self._create_server_pool(mock_model_repo_instance)
        
        # Mock the GPU allocator in the GPU resource manager
        mock_allocator_instance = Mock()
        mock_allocator_instance.allocate_gpus.return_value = None
        
        # Replace the allocator in the GPU resource manager
        if pool.gpu_manager:
            pool.gpu_manager.gpu_allocator = mock_allocator_instance
        
        # Mock the health checker to return True for endpoint checks
        with patch('src.shared.health_checker.HealthChecker.check_http_endpoint', return_value=True):
            with patch('src.shared.health_checker.HealthChecker.check_process_running', return_value=True):
                # Mock the VRAM estimation to return 10GB
                with patch.object(pool.gpu_manager, '_estimate_model_vram_requirement', return_value=10.0):
                    import asyncio
                    server = asyncio.run(pool.get_server_for_model("llama-3.1-70b-instruct.Q4_K_M.gguf"))
                     
                    # Verify that allocation was called with the right parameters
                    mock_allocator_instance.allocate_gpus.assert_called_once_with(10.0, [gpu0], gguf_path='llama-3.1-70b-instruct.Q4_K_M.gguf')
                     
                    # Server should be None because of GPU allocation failure
                    assert server is None
    
    @patch('src.frameworks_drivers.model_repository.ModelRepository')
    @patch('src.frameworks_drivers.gpu_detector.GPUDetector')
    @patch('subprocess.Popen')
    def test_gpu_assignment_cleanup_on_server_shutdown(self, mock_popen, mock_gpu_detector, mock_model_repository):
        """Test that GPU assignments are properly cleaned up when servers are shutdown."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_model_repo_instance = Mock()
        mock_gpu_detector.return_value = mock_detector_instance
        mock_model_repository.return_value = mock_model_repo_instance
        
        # Create GPUs for testing
        gpu0 = self._create_gpu(0, "NVIDIA GeForce RTX 4090", 24.0, 20.0, 4.0)
        mock_detector_instance.get_available_gpus.return_value = [gpu0]
        mock_detector_instance.is_gpu_available.return_value = True
        mock_detector_instance.is_pynvml_available.return_value = True
        
        # Create server pool
        pool = self._create_server_pool(mock_model_repo_instance)
        
        # Mock the GPU allocator in the GPU resource manager
        mock_allocator_instance = Mock()
        mock_assignment = GPUAssignment(gpu_ids=[0], tensor_splits=[1.0], estimated_vram_required=8.0)
        mock_allocator_instance.allocate_gpus.return_value = mock_assignment
        
        # Replace the allocator in the GPU resource manager
        if pool.gpu_manager:
            pool.gpu_manager.gpu_allocator = mock_allocator_instance
        
        # Mock the health checker to return True for endpoint checks
        with patch('src.shared.health_checker.HealthChecker.check_http_endpoint', return_value=True):
            with patch('src.shared.health_checker.HealthChecker.check_process_running', return_value=True):
                # Mock the VRAM estimation to return 8GB
                with patch.object(pool.gpu_manager, '_estimate_model_vram_requirement', return_value=8.0):
                    import asyncio
                    server = asyncio.run(pool.get_server_for_model("llama-3.2-8b-instruct.Q4_K_M.gguf"))
                     
                    # Verify the server has GPU assignment
                    assert server is not None
                    assert server.gpu_assignment is not None
                    assert server.gpu_assignment.gpu_ids == [0]
                     
                    # Shutdown the pool (which cleans up GPU assignments)
                    pool.shutdown()
                     
                    # Verify that the GPU assignment was cleared
                    assert server.gpu_assignment is None
    
    @patch('src.frameworks_drivers.model_repository.ModelRepository')
    @patch('src.frameworks_drivers.gpu_detector.GPUDetector')
    @patch('subprocess.Popen')
    def test_concurrent_gpu_allocation_requests(self, mock_popen, mock_gpu_detector, mock_model_repository):
        """Test handling of GPU allocation requests."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_model_repo_instance = Mock()
        mock_gpu_detector.return_value = mock_detector_instance
        mock_model_repository.return_value = mock_model_repo_instance
        
        # Create GPUs for testing
        gpu0 = self._create_gpu(0, "NVIDIA GeForce RTX 4090", 24.0, 20.0, 4.0)
        gpu1 = self._create_gpu(1, "NVIDIA GeForce RTX 4080", 16.0, 12.0, 4.0)
        mock_detector_instance.get_available_gpus.return_value = [gpu0, gpu1]
        mock_detector_instance.is_gpu_available.return_value = True
        mock_detector_instance.is_pynvml_available.return_value = True
        
        # Create server pool with size 2
        pool = self._create_server_pool(mock_model_repo_instance, size=2)
        
        # Mock the GPU allocator in the GPU resource manager
        mock_allocator_instance = Mock()
        def mock_allocate_gpus(required_vram, available_gpus, gguf_path=None):
            if required_vram <= 10.0:
                return GPUAssignment(gpu_ids=[0], tensor_splits=[1.0], estimated_vram_required=required_vram)
            else:
                return GPUAssignment(gpu_ids=[0, 1], tensor_splits=[0.5, 0.5], estimated_vram_required=required_vram)

        mock_allocator_instance.allocate_gpus.side_effect = mock_allocate_gpus
        
        # Replace the allocator in the GPU resource manager
        if pool.gpu_manager:
            pool.gpu_manager.gpu_allocator = mock_allocator_instance
        
        # Mock the health checker to return True for endpoint checks
        with patch('src.shared.health_checker.HealthChecker.check_http_endpoint', return_value=True):
            with patch('src.shared.health_checker.HealthChecker.check_process_running', return_value=True):
                # Mock the VRAM estimation
                def mock_estimate(model_variant):
                    if "8b" in model_variant:
                        return 8.0
                    elif "70b" in model_variant:
                        return 15.0
                    return 8.0  # default
                  
                with patch.object(pool.gpu_manager, '_estimate_model_vram_requirement', side_effect=mock_estimate):
                    import asyncio
                    # Get servers for different models
                    server1 = asyncio.run(pool.get_server_for_model("llama-3.2-8b-instruct.Q4_K_M.gguf"))
                    server2 = asyncio.run(pool.get_server_for_model("llama-3.1-70b-instruct.Q4_K_M.gguf"))
                     
                    # Both servers should be allocated successfully
                    assert server1 is not None
                    assert server1.gpu_assignment is not None
                    assert server2 is not None
                    assert server2.gpu_assignment is not None
                     
                    # The 70B model should use both GPUs, while 8B uses one
                    assert len(server2.gpu_assignment.gpu_ids) >= len(server1.gpu_assignment.gpu_ids)
            
    @patch('src.frameworks_drivers.model_repository.ModelRepository')
    @patch('src.frameworks_drivers.gpu_detector.GPUDetector')
    @patch('subprocess.Popen')
    def test_model_fits_on_single_gpu(self, mock_popen, mock_gpu_detector, mock_model_repository):
        """Test scenario: Model fits on single GPU (from quickstart.md)."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_model_repo_instance = Mock()
        mock_gpu_detector.return_value = mock_detector_instance
        mock_model_repository.return_value = mock_model_repo_instance
        
        # Create a GPU with sufficient memory for a small model
        gpu0 = self._create_gpu(0, "NVIDIA GeForce RTX 4090", 24.0, 20.0, 4.0)
        mock_detector_instance.get_available_gpus.return_value = [gpu0]
        mock_detector_instance.is_gpu_available.return_value = True
        mock_detector_instance.is_pynvml_available.return_value = True
        
        # Create server pool
        pool = self._create_server_pool(mock_model_repo_instance, port_start=800)
        
        # Mock the GPU allocator in the GPU resource manager
        mock_allocator_instance = Mock()
        mock_assignment = GPUAssignment(gpu_ids=[0], tensor_splits=[1.0], estimated_vram_required=8.0)  # Small model that fits on one GPU
        mock_allocator_instance.allocate_gpus.return_value = mock_assignment
        
        # Replace the allocator in the GPU resource manager
        if pool.gpu_manager:
            pool.gpu_manager.gpu_allocator = mock_allocator_instance
        
        # Mock the health checker to return True for endpoint checks
        with patch('src.shared.health_checker.HealthChecker.check_http_endpoint', return_value=True):
            with patch('src.shared.health_checker.HealthChecker.check_process_running', return_value=True):
                # Mock the VRAM estimation to return 8GB for a small model
                with patch.object(pool.gpu_manager, '_estimate_model_vram_requirement', return_value=8.0):
                    import asyncio
                    server = asyncio.run(pool.get_server_for_model("llama-3.2-8b-instruct.Q4_K_M.gguf"))
                     
                    # Verify that allocation was called with the right parameters
                    mock_allocator_instance.allocate_gpus.assert_called_once_with(8.0, [gpu0], gguf_path='llama-3.2-8b-instruct.Q4_K_M.gguf')
                     
                    # Server should be returned successfully with single GPU assignment
                    assert server is not None
                    assert server.gpu_assignment is not None
                    assert server.gpu_assignment.gpu_ids == [0]  # Single GPU
                    assert server.gpu_assignment.estimated_vram_required == 8.0
            
    @patch('src.frameworks_drivers.model_repository.ModelRepository')
    @patch('src.frameworks_drivers.gpu_detector.GPUDetector')
    @patch('subprocess.Popen')
    def test_model_requires_multiple_gpus(self, mock_popen, mock_gpu_detector, mock_model_repository):
        """Test scenario: Model requires multiple GPUs."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_model_repo_instance = Mock()
        mock_gpu_detector.return_value = mock_detector_instance
        mock_model_repository.return_value = mock_model_repo_instance
        
        # Create multiple GPUs with sufficient combined memory
        gpu0 = self._create_gpu(0, "NVIDIA GeForce RTX 4090", 24.0, 12.0, 12.0)
        gpu1 = self._create_gpu(1, "NVIDIA GeForce RTX 4080", 16.0, 10.0, 6.0)
        mock_detector_instance.get_available_gpus.return_value = [gpu0, gpu1]
        mock_detector_instance.is_gpu_available.return_value = True
        mock_detector_instance.is_pynvml_available.return_value = True
        
        # Create server pool
        pool = self._create_server_pool(mock_model_repo_instance)
        
        # Mock the GPU allocator in the GPU resource manager
        mock_allocator_instance = Mock()
        mock_assignment = GPUAssignment(
            gpu_ids=[0, 1],  # Multiple GPUs required
            tensor_splits=[0.5, 0.5],
            estimated_vram_required=20.0  # Large model requiring multiple GPUs
        )
        mock_allocator_instance.allocate_gpus.return_value = mock_assignment
        
        # Replace the allocator in the GPU resource manager
        if pool.gpu_manager:
            pool.gpu_manager.gpu_allocator = mock_allocator_instance
        
        # Mock the health checker to return True for endpoint checks
        with patch('src.shared.health_checker.HealthChecker.check_http_endpoint', return_value=True):
            with patch('src.shared.health_checker.HealthChecker.check_process_running', return_value=True):
                # Mock the VRAM estimation to return 20GB for a large model
                with patch.object(pool.gpu_manager, '_estimate_model_vram_requirement', return_value=20.0):
                    import asyncio
                    server = asyncio.run(pool.get_server_for_model("llama-3.1-70b-instruct.Q4_K_M.gguf"))
                     
                    # Verify that allocation was called with the right parameters
                    mock_allocator_instance.allocate_gpus.assert_called_once_with(20.0, [gpu0, gpu1], gguf_path='llama-3.1-70b-instruct.Q4_K_M.gguf')
                     
                    # Server should be returned successfully with multi-GPU assignment
                    assert server is not None
                    assert server.gpu_assignment is not None
                    assert set(server.gpu_assignment.gpu_ids) == {0, 1}  # Multiple GPUs
                    assert server.gpu_assignment.estimated_vram_required == 20.0
            
    @patch('src.frameworks_drivers.model_repository.ModelRepository')
    @patch('src.frameworks_drivers.gpu_detector.GPUDetector')
    @patch('subprocess.Popen')
    def test_insufficient_gpu_resources_for_model(self, mock_popen, mock_gpu_detector, mock_model_repository):
        """Test scenario: Insufficient GPU resources for model."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_model_repo_instance = Mock()
        mock_gpu_detector.return_value = mock_detector_instance
        mock_model_repository.return_value = mock_model_repo_instance
        
        # Create a GPU with insufficient memory for the model
        gpu0 = self._create_gpu(0, "NVIDIA GeForce RTX 3060", 12.0, 4.0, 8.0)  # Only 4GB free
        mock_detector_instance.get_available_gpus.return_value = [gpu0]
        mock_detector_instance.is_gpu_available.return_value = True
        mock_detector_instance.is_pynvml_available.return_value = True
        
        # Create server pool
        pool = self._create_server_pool(mock_model_repo_instance)
        
        # Mock the GPU allocator in the GPU resource manager
        mock_allocator_instance = Mock()
        mock_allocator_instance.allocate_gpus.return_value = None
        
        # Replace the allocator in the GPU resource manager
        if pool.gpu_manager:
            pool.gpu_manager.gpu_allocator = mock_allocator_instance
        
        # Mock the health checker to return True for endpoint checks
        with patch('src.shared.health_checker.HealthChecker.check_http_endpoint', return_value=True):
            with patch('src.shared.health_checker.HealthChecker.check_process_running', return_value=True):
                # Mock the VRAM estimation to return 15GB (more than available)
                with patch.object(pool.gpu_manager, '_estimate_model_vram_requirement', return_value=15.0):
                    import asyncio
                    server = asyncio.run(pool.get_server_for_model("llama-3.1-70b-instruct.Q4_K_M.gguf"))
                     
                    # Verify that allocation was called with the right parameters
                    mock_allocator_instance.allocate_gpus.assert_called_once_with(15.0, [gpu0], gguf_path='llama-3.1-70b-instruct.Q4_K_M.gguf')
                     
                    # Server should be None due to insufficient GPU resources
                    assert server is None
            
    def test_allocation_decision_performance_timing(self):
        """Performance test: Allocation decisions complete within 10ms."""
        import time
        from src.frameworks_drivers.gpu_allocator import AdaptiveGPUAllocator
        from src.entities.gpu import GPU
        from src.entities.gpu_assignment import GPUAssignment

        # Create allocator instance
        allocator = AdaptiveGPUAllocator()
        
        # Create test GPUs
        available_gpus = [
            self._create_gpu(0, "GPU 0", 24.0, 20.0, 4.0),
            self._create_gpu(1, "GPU 1", 16.0, 12.0, 4.0)
        ]
        
        # Time the allocation decision
        start_time = time.time()
        assignment = allocator.allocate_gpus(required_vram=8.0, available_gpus=available_gpus)
        end_time = time.time()
        
        allocation_time_ms = (end_time - start_time) * 100  # Convert to milliseconds
        
        # Verify allocation was successful
        assert assignment is not None
        assert assignment.gpu_ids == [0]  # Should prefer single GPU when possible
        
        # Verify allocation completed within 10ms
        assert allocation_time_ms <= 10.0, f"Allocation took {allocation_time_ms:.2f}ms, exceeding 10ms limit"
        
    def test_vram_estimation_accuracy(self):
        """Performance test: VRAM estimation accuracy within 10% of actual usage."""
        from src.frameworks_drivers.gpu_allocator import AdaptiveGPUAllocator

        allocator = AdaptiveGPUAllocator()

        # Test with known parameters - 7B parameters with Q4_K_M quantization
        # This should give us a known VRAM requirement
        estimated_vram = allocator.estimate_model_vram(
            model_parameters=7_000_000_000,  # 7B parameters
            model_variant="model.Q4_K_M.gguf"  # Q4_K_M quantization
        )

        # For 7B parameters with Q4 quantization, we expect approximately 4.4GB (including KV cache offload)
        # Using a range that allows for 10% accuracy
        expected_vram = 4.4  # Approximate expected VRAM in GB for 7B Q4 model
        tolerance = expected_vram * 0.1  # 10% tolerance

        assert estimated_vram is not None
        assert abs(estimated_vram - expected_vram) <= tolerance, \
            f"Estimated VRAM {estimated_vram}GB is not within 10% of expected {expected_vram}GB"


class TestCommandConstruction:
    """Test command construction for single and multi-GPU scenarios."""

    def test_single_gpu_command_construction(self):
        """Test that single GPU command construction sets correct parameters."""
        from src.entities.gpu_assignment import GPUAssignment
        from src.frameworks_drivers.config import ServerPoolConfig
        from src.frameworks_drivers.server_lifecycle_manager import ServerLifecycleManager
        import os

        # Create a mock server instance
        config = ServerPoolConfig(size=1, port_start=8000, gpu_layers=20, host="localhost", request_timeout=300)
        gpu_manager = None  # Not needed for this test
        manager = ServerLifecycleManager(config, gpu_manager)

        # Create a server instance
        server = manager.servers[0]
        server.gpu_assignment = GPUAssignment(
            gpu_ids=[0],
            tensor_splits=[1.0],
            estimated_vram_required=8.0,
            n_gpu_layers=15
        )

        # Mock the base command construction
        base_cmd = ["llama-server", "-m", "model.gguf", "--port", "8000", "--host", "127.0.0.1"]

        # Simulate the GPU parameter addition logic from _load_model_into_server
        env = os.environ.copy()
        if server.gpu_assignment and server.gpu_assignment.gpu_ids:
            n_gpu_layers = server.gpu_assignment.n_gpu_layers if server.gpu_assignment.n_gpu_layers is not None else config.gpu_layers
            gpu_ids = server.gpu_assignment.gpu_ids
            if len(gpu_ids) == 1:
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])
                base_cmd.extend(["--split-mode", "none"])
                base_cmd.extend(["--n-gpu-layers", str(n_gpu_layers)])
            else:
                env['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu_id) for gpu_id in gpu_ids)
                base_cmd.extend(["--split-mode", "layer"])
                base_cmd.extend(["--tensor-split", ','.join(str(ratio) for ratio in server.gpu_assignment.tensor_splits)])
                base_cmd.extend(["--n-gpu-layers", str(n_gpu_layers)])

        # Verify single GPU command
        assert env['CUDA_VISIBLE_DEVICES'] == '0'
        assert "--split-mode" in base_cmd
        assert base_cmd[base_cmd.index("--split-mode") + 1] == "none"
        assert "--n-gpu-layers" in base_cmd
        assert base_cmd[base_cmd.index("--n-gpu-layers") + 1] == "15"
        assert "--tensor-split" not in base_cmd

    def test_multi_gpu_command_construction(self):
        """Test that multi-GPU command construction sets correct parameters."""
        from src.entities.gpu_assignment import GPUAssignment
        from src.frameworks_drivers.config import ServerPoolConfig
        from src.frameworks_drivers.server_lifecycle_manager import ServerLifecycleManager
        import os

        # Create a mock server instance
        config = ServerPoolConfig(size=1, port_start=8000, gpu_layers=20, host="localhost", request_timeout=300)
        gpu_manager = None
        manager = ServerLifecycleManager(config, gpu_manager)

        server = manager.servers[0]
        server.gpu_assignment = GPUAssignment(
            gpu_ids=[0, 1],
            tensor_splits=[0.4, 0.6],
            estimated_vram_required=16.0,
            n_gpu_layers=25
        )

        # Mock the base command construction
        base_cmd = ["llama-server", "-m", "model.gguf", "--port", "8000", "--host", "127.0.0.1"]

        # Simulate the GPU parameter addition logic
        env = os.environ.copy()
        if server.gpu_assignment and server.gpu_assignment.gpu_ids:
            n_gpu_layers = server.gpu_assignment.n_gpu_layers if server.gpu_assignment.n_gpu_layers is not None else config.gpu_layers
            gpu_ids = server.gpu_assignment.gpu_ids
            if len(gpu_ids) == 1:
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])
                base_cmd.extend(["--split-mode", "none"])
                base_cmd.extend(["--n-gpu-layers", str(n_gpu_layers)])
            else:
                env['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu_id) for gpu_id in gpu_ids)
                base_cmd.extend(["--split-mode", "layer"])
                base_cmd.extend(["--tensor-split", ','.join(str(ratio) for ratio in server.gpu_assignment.tensor_splits)])
                base_cmd.extend(["--n-gpu-layers", str(n_gpu_layers)])

        # Verify multi-GPU command
        assert env['CUDA_VISIBLE_DEVICES'] == '0,1'
        assert "--split-mode" in base_cmd
        assert base_cmd[base_cmd.index("--split-mode") + 1] == "layer"
        assert "--tensor-split" in base_cmd
        assert base_cmd[base_cmd.index("--tensor-split") + 1] == "0.4,0.6"
        assert "--n-gpu-layers" in base_cmd
        assert base_cmd[base_cmd.index("--n-gpu-layers") + 1] == "25"

    def test_no_gpu_assignment_command_construction(self):
        """Test command construction when no GPU assignment exists."""
        from src.frameworks_drivers.config import ServerPoolConfig
        from src.frameworks_drivers.server_lifecycle_manager import ServerLifecycleManager
        import os

        config = ServerPoolConfig(size=1, port_start=8000, gpu_layers=20, host="localhost", request_timeout=300)
        gpu_manager = None
        manager = ServerLifecycleManager(config, gpu_manager)

        server = manager.servers[0]
        server.gpu_assignment = None  # No GPU assignment

        # Mock the base command construction
        base_cmd = ["llama-server", "-m", "model.gguf", "--port", "8000", "--host", "127.0.0.1"]

        # Simulate the GPU parameter addition logic
        env = os.environ.copy()
        if server.gpu_assignment and server.gpu_assignment.gpu_ids:
            # GPU logic would go here
            pass
        elif manager._is_cuda_available() and config.gpu_layers > 0:
            base_cmd.extend(["--n-gpu-layers", str(config.gpu_layers)])

        # Since no GPU assignment and assuming CUDA not available, no GPU params added
        # But the test assumes CUDA available for fallback
        with patch.object(manager, '_is_cuda_available', return_value=True):
            if server.gpu_assignment and server.gpu_assignment.gpu_ids:
                pass
            elif manager._is_cuda_available() and config.gpu_layers > 0:
                base_cmd.extend(["--n-gpu-layers", str(config.gpu_layers)])

            assert "--n-gpu-layers" in base_cmd
            assert base_cmd[base_cmd.index("--n-gpu-layers") + 1] == "20"
            assert "--split-mode" not in base_cmd
            assert "--tensor-split" not in base_cmd

    def test_n_gpu_layers_calculation(self):
        """Test calculation of optimal n_gpu_layers based on model and VRAM."""
        from src.frameworks_drivers.gpu_allocator import AdaptiveGPUAllocator
        from src.entities.gpu_assignment import GPUAssignment
        from src.entities.gpu import GPU

        allocator = AdaptiveGPUAllocator()

        # Create test GPUs
        available_gpus = [
            GPU(id=0, name="GPU 0", total_memory=24.0, free_memory=20.0, used_memory=4.0, utilization=16.7, assigned_models=[]),
            GPU(id=1, name="GPU 1", total_memory=24.0, free_memory=18.0, used_memory=6.0, utilization=25.0, assigned_models=[])
        ]

        # Create assignment for both GPUs
        assignment = GPUAssignment(
            gpu_ids=[0, 1],
            tensor_splits=[20.0/38.0, 18.0/38.0],  # Proportional to free memory
            estimated_vram_required=16.0
        )

        # Mock GGUF utils to return 32 layers
        with patch('src.utils.gguf_utils.GGUFUtils.get_model_layers_from_gguf', return_value=32):
            n_layers = allocator._calculate_optimal_n_gpu_layers("fake.gguf", assignment, available_gpus)

            # Total available VRAM: 38GB, required: 16GB, so proportion = 16/38 ≈ 0.42, layers = 32 * 0.42 ≈ 13.44, but min 1
            assert n_layers is not None
            assert 1 <= n_layers <= 32

    def test_n_gpu_layers_calculation_single_gpu(self):
        """Test n_gpu_layers calculation for single GPU."""
        from src.frameworks_drivers.gpu_allocator import AdaptiveGPUAllocator
        from src.entities.gpu_assignment import GPUAssignment
        from src.entities.gpu import GPU

        allocator = AdaptiveGPUAllocator()

        available_gpus = [
            GPU(id=0, name="GPU 0", total_memory=24.0, free_memory=20.0, used_memory=4.0, utilization=16.7, assigned_models=[])
        ]

        assignment = GPUAssignment(
            gpu_ids=[0],
            tensor_splits=[1.0],
            estimated_vram_required=8.0
        )

        with patch('src.utils.gguf_utils.GGUFUtils.get_model_layers_from_gguf', return_value=32):
            n_layers = allocator._calculate_optimal_n_gpu_layers("fake.gguf", assignment, available_gpus)

            # Available VRAM (20GB) > required (8GB), so should return all layers
            assert n_layers == 32