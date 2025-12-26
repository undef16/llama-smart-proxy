"""Tests for CPU fallback functionality when GPU is not available."""
import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.frameworks_drivers.config import ServerPoolConfig
from src.entities.gpu_assignment import GPUAssignment


def test_gpu_detector_without_pynvml():
    """Test GPU detector gracefully handles missing pynvml library."""
    # Mock pynvml in the gpu_detector module to None
    with patch('src.frameworks_drivers.gpu_detector.pynvml', None):
        from src.frameworks_drivers.gpu_detector import GPUDetector
        detector = GPUDetector()
        gpus = detector.detect_gpus()
        assert gpus == []
        

def test_gpu_detector_with_pynvml_but_no_gpus():
    """Test GPU detector when pynvml is available but no GPUs are present."""
    # Import pynvml inside the test to handle import errors gracefully
    try:
        import pynvml  # This will import either nvidia-ml-py or pynvml (both use same import)
    except ImportError:
        # If pynvml is not available, skip this test
        pytest.skip("pynvml/nvidia-ml-py not available, skipping test")
        return

    # Mock pynvml to simulate no GPUs
    with patch.object(pynvml, 'nvmlInit', side_effect=pynvml.NVMLError(1)):
        from src.frameworks_drivers.gpu_detector import GPUDetector
        detector = GPUDetector()
        gpus = detector.detect_gpus()
        assert gpus == []
        assert not detector.is_gpu_available()


def test_server_pool_cpu_only_fallback():
    """Test that server pool falls back to CPU-only operation when no GPUs are available."""
    config = ServerPoolConfig(size=2, host="localhost", port_start=8001, gpu_layers=0, request_timeout=300)
    
    # Create a mock model repository for VRAM estimation
    mock_model_repository = Mock()
    mock_model_repository.get_model.return_value = {
        'id': 'test-model',
        'repo': 'test/repo',
        'variant': 'test-variant',
        'backend': 'llama.cpp'
    }
    
    # Import and create server pool
    from src.frameworks_drivers.server_pool import ServerPool
    pool = ServerPool(config, model_repository=mock_model_repository)
    
    # Mock the GPU resource manager to simulate no GPUs available
    pool.gpu_manager = Mock()
    pool.gpu_manager.gpu_detector = Mock()
    pool.gpu_manager.gpu_detector.get_available_gpus.return_value = []
    pool.gpu_manager.gpu_detector.is_gpu_available.return_value = False
    pool.gpu_manager.gpu_detector.is_pynvml_available.return_value = True  # pynvml is available but no GPUs

    # Mock the GPU allocator to return None (no allocation)
    pool.gpu_manager.gpu_allocator = Mock()
    pool.gpu_manager.gpu_allocator.allocate_gpus.return_value = None
    pool.gpu_manager.gpu_allocator.estimate_model_vram.return_value = 2.0  # 2GB estimated requirement

    # Mock allocate_gpu_for_model to return None
    pool.gpu_manager.allocate_gpu_for_model.return_value = None
    
    # Use the test helper method to avoid complex subprocess mocking
    with patch.object(pool.server_manager, '_load_model_into_server', side_effect=pool._test_load_model_success):
        # Run the async function
        result = asyncio.run(pool.get_server_for_model('test-model'))
        
        # Should succeed in getting a server even without GPUs
        assert result is not None
        # GPU assignment should be None
        assert result.gpu_assignment is None


def test_server_pool_with_pynvml_unavailable():
    """Test that server pool works when pynvml is not available."""
    config = ServerPoolConfig(size=2, host="localhost", port_start=8001, gpu_layers=0, request_timeout=300)
    
    # Create a mock model repository for VRAM estimation
    mock_model_repository = Mock()
    mock_model_repository.get_model.return_value = {
        'id': 'test-model',
        'repo': 'test/repo',
        'variant': 'test-variant',
        'backend': 'llama.cpp'
    }

    # Import and create server pool
    from src.frameworks_drivers.server_pool import ServerPool
    pool = ServerPool(config, model_repository=mock_model_repository)

    # Mock the GPU resource manager to simulate pynvml not being available
    pool.gpu_manager = Mock()
    pool.gpu_manager.gpu_detector = Mock()
    pool.gpu_manager.gpu_detector.get_available_gpus.return_value = []
    pool.gpu_manager.gpu_detector.is_gpu_available.return_value = False
    pool.gpu_manager.gpu_detector.is_pynvml_available.return_value = False  # pynvml is not available

    # Mock the GPU allocator to return None (no allocation)
    pool.gpu_manager.gpu_allocator = Mock()
    pool.gpu_manager.gpu_allocator.allocate_gpus.return_value = None
    pool.gpu_manager.gpu_allocator.estimate_model_vram.return_value = 2.0  # 2GB estimated requirement

    # Mock allocate_gpu_for_model to return None
    pool.gpu_manager.allocate_gpu_for_model.return_value = None

    # Use the test helper method to avoid complex subprocess mocking
    with patch.object(pool.server_manager, '_load_model_into_server', side_effect=pool._test_load_model_success):
        # Run the async function
        result = asyncio.run(pool.get_server_for_model('test-model'))

        # Should succeed in getting a server even without GPUs
        assert result is not None
        # GPU assignment should be None since no GPUs are available
        assert result.gpu_assignment is None


def test_gpu_allocation_skipped_when_no_gpus():
    """Test that GPU allocation is skipped when no GPUs are available."""
    config = ServerPoolConfig(size=1, host="localhost", port_start=8001, gpu_layers=20, request_timeout=300)
    
    # Create a mock model repository for VRAM estimation
    mock_model_repository = Mock()
    mock_model_repository.get_model.return_value = {
        'id': 'test-model',
        'repo': 'test/repo',
        'variant': 'test-variant',
        'backend': 'llama.cpp'
    }
    
    # Import and create server pool
    from src.frameworks_drivers.server_pool import ServerPool
    pool = ServerPool(config, model_repository=mock_model_repository)
    
    # Mock the GPU resource manager to simulate no GPUs available
    pool.gpu_manager = Mock()
    pool.gpu_manager.gpu_detector = Mock()
    pool.gpu_manager.gpu_detector.get_available_gpus.return_value = []
    pool.gpu_manager.gpu_detector.is_gpu_available.return_value = False
    pool.gpu_manager.gpu_detector.is_pynvml_available.return_value = True

    # Mock the GPU allocator
    pool.gpu_manager.gpu_allocator = Mock()
    pool.gpu_manager.gpu_allocator.allocate_gpus.return_value = None  # Should not be called
    pool.gpu_manager.gpu_allocator.estimate_model_vram.return_value = 2.0

    # Mock allocate_gpu_for_model to return None
    pool.gpu_manager.allocate_gpu_for_model.return_value = None
    
    # Use the test helper method to avoid complex subprocess mocking
    with patch.object(pool.server_manager, '_load_model_into_server', side_effect=pool._test_load_model_success):
        # Run the async function
        result = asyncio.run(pool.get_server_for_model('test-model'))
        
        # Should succeed in getting a server
        assert result is not None
        # GPU allocation should not have been attempted
        pool.gpu_manager.gpu_allocator.allocate_gpus.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])