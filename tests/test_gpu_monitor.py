"""
Tests for GPU monitoring functionality.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.entities.gpu import GPU
from src.frameworks_drivers.gpu_monitor import GPUMonitor


class TestGPUMonitor:
    """Test cases for GPUMonitor class."""
    
    def test_gpu_monitor_initialization_success(self):
        """Test successful initialization of GPU monitor."""
        with patch('src.frameworks_drivers.gpu_monitor.pynvml') as mock_pynvml:
            # Mock successful initialization
            mock_pynvml.nvmlInit = Mock()
            mock_pynvml.NVMLError = Exception  # Mock the exception class
            
            monitor = GPUMonitor()
            
            assert monitor.initialized is True
            mock_pynvml.nvmlInit.assert_called_once()
    
    def test_gpu_monitor_initialization_failure(self):
        """Test GPU monitor initialization failure."""
        with patch('src.frameworks_drivers.gpu_monitor.pynvml') as mock_pynvml:
            # Mock initialization failure
            mock_pynvml.nvmlInit.side_effect = Exception("Initialization failed")
            mock_pynvml.NVMLError = Exception  # Mock the exception class
            
            monitor = GPUMonitor()
            
            assert monitor.initialized is False
    
    def test_get_gpu_count_initialized(self):
        """Test getting GPU count when initialized."""
        with patch('src.frameworks_drivers.gpu_monitor.pynvml') as mock_pynvml:
            mock_pynvml.nvmlInit = Mock()
            mock_pynvml.nvmlDeviceGetCount = Mock(return_value=2)
            mock_pynvml.NVMLError = Exception
            
            monitor = GPUMonitor()
            
            count = monitor.get_gpu_count()
            assert count == 2
            mock_pynvml.nvmlDeviceGetCount.assert_called_once()
    
    def test_get_gpu_count_not_initialized(self):
        """Test getting GPU count when not initialized."""
        with patch('src.frameworks_drivers.gpu_monitor.pynvml') as mock_pynvml:
            mock_pynvml.nvmlInit = Mock(side_effect=Exception())
            mock_pynvml.NVMLError = Exception
            
            monitor = GPUMonitor()
            
            count = monitor.get_gpu_count()
            assert count == 0
    
    def test_get_gpu_info_success(self):
        """Test getting GPU info successfully."""
        with patch('src.frameworks_drivers.gpu_monitor.pynvml') as mock_pynvml:
            mock_pynvml.nvmlInit = Mock()
            mock_handle = Mock()
            mock_pynvml.nvmlDeviceGetHandleByIndex = Mock(return_value=mock_handle)
            mock_pynvml.nvmlDeviceGetName = Mock(return_value=b"Test GPU")
            mock_pynvml.nvmlDeviceGetMemoryInfo = Mock(return_value=Mock(total=8*1024**3, used=2*1024**3, free=6*1024**3))
            mock_pynvml.nvmlDeviceGetUtilizationRates = Mock(return_value=Mock(gpu=50))
            mock_pynvml.nvmlDeviceGetTemperature = Mock(return_value=65)
            mock_pynvml.nvmlDeviceGetPowerUsage = Mock(return_value=100000)  # 100W
            mock_pynvml.nvmlDeviceGetCudaComputeCapability = Mock(return_value=(7, 5))
            mock_pynvml.NVMLError = Exception
            
            monitor = GPUMonitor()
            gpu_info = monitor.get_gpu_info(0)
            
            assert gpu_info is not None
            assert gpu_info.id == 0
            assert gpu_info.name == "Test GPU"
            assert gpu_info.total_memory == 8.0
            assert gpu_info.used_memory == 2.0
            assert gpu_info.free_memory == 6.0
            assert gpu_info.utilization == 50
            assert gpu_info.temperature == 65
            assert gpu_info.power_usage == 100.0 # 100W
            assert gpu_info.compute_capability == "7.5"
    
    def test_get_gpu_info_failure(self):
        """Test getting GPU info when pynvml operations fail."""
        with patch('src.frameworks_drivers.gpu_monitor.pynvml') as mock_pynvml:
            mock_pynvml.nvmlInit = Mock()
            mock_pynvml.nvmlDeviceGetHandleByIndex = Mock(side_effect=Exception("Device error"))
            mock_pynvml.NVMLError = Exception
            
            monitor = GPUMonitor()
            gpu_info = monitor.get_gpu_info(0)
            
            assert gpu_info is None
    
    def test_get_all_gpus(self):
        """Test getting all GPUs."""
        with patch('src.frameworks_drivers.gpu_monitor.pynvml') as mock_pynvml:
            mock_pynvml.nvmlInit = Mock()
            mock_pynvml.nvmlDeviceGetCount = Mock(return_value=2)
            mock_pynvml.nvmlDeviceGetHandleByIndex = Mock()
            mock_pynvml.nvmlDeviceGetName = Mock(return_value=b"Test GPU")
            mock_pynvml.nvmlDeviceGetMemoryInfo = Mock(return_value=Mock(total=8*1024**3, used=2*1024**3, free=6*1024**3))
            mock_pynvml.nvmlDeviceGetUtilizationRates = Mock(return_value=Mock(gpu=50))
            mock_pynvml.nvmlDeviceGetTemperature = Mock(return_value=65)
            mock_pynvml.nvmlDeviceGetPowerUsage = Mock(return_value=100000)
            mock_pynvml.nvmlDeviceGetCudaComputeCapability = Mock(return_value=(7, 5))
            mock_pynvml.NVMLError = Exception
            
            monitor = GPUMonitor()
            gpus = monitor.get_all_gpus()
            
            assert len(gpus) == 2
            for i, gpu in enumerate(gpus):
                assert gpu.id == i
                assert gpu.name == "Test GPU"
                assert gpu.total_memory == 8.0


class MockGPUEnvironment:
    """Mock environment for GPU testing."""
    
    def __init__(self):
        self.gpus = [
            GPU(
                id=0,
                name="Mock GPU 0",
                total_memory=8.0,
                free_memory=6.0,
                used_memory=2.0,
                utilization=25.0,
                temperature=45.0,
                power_usage=75.0,
                assigned_models=[],
                compute_capability="7.5"
            ),
            GPU(
                id=1,
                name="Mock GPU 1",
                total_memory=8.0,
                free_memory=3.0,
                used_memory=5.0,
                utilization=60.0,
                temperature=60.0,
                power_usage=120.0,
                assigned_models=[],
                compute_capability="7.5"
            )
        ]
    
    def get_mock_gpus(self):
        """Get mock GPU data."""
        return self.gpus


def test_mock_gpu_environment():
    """Test the mock GPU environment."""
    mock_env = MockGPUEnvironment()
    gpus = mock_env.get_mock_gpus()
    
    assert len(gpus) == 2
    assert gpus[0].id == 0
    assert gpus[0].free_memory == 6.0
    assert gpus[1].id == 1
    assert gpus[1].free_memory == 3.0