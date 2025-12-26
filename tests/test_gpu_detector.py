"""
Tests for GPU detection functionality.
"""
import pytest
from unittest.mock import Mock, patch

from src.entities.gpu import GPU
from src.frameworks_drivers.gpu_detector import GPUDetector


class TestGPUDetector:
    """Test cases for GPUDetector class."""
    
    def test_gpu_detector_initialization(self):
        """Test GPU detector initialization."""
        detector = GPUDetector()
        
        assert detector is not None
        assert detector.detected_gpus == []
        assert detector.gpu_available is False
    
    @patch('src.frameworks_drivers.gpu_detector.pynvml')
    def test_detect_gpus_success(self, mock_pynvml):
        """Test successful GPU detection."""
        # Mock pynvml functions
        mock_pynvml.nvmlInit = Mock()
        mock_pynvml.nvmlShutdown = Mock()
        mock_pynvml.NVMLError = Exception  # Mock the exception class
        
        # Mock GPU monitor to return GPUs
        mock_gpu_0 = GPU(
            id=0,
            name="Test GPU 0",
            total_memory=8.0,
            free_memory=6.0,
            used_memory=2.0,
            utilization=25.0,
            assigned_models=[]
        )
        mock_gpu_1 = GPU(
            id=1,
            name="Test GPU 1",
            total_memory=8.0,
            free_memory=4.0,
            used_memory=4.0,
            utilization=50.0,
            assigned_models=[]
        )
        
        # Mock the gpu_monitor.get_all_gpus method
        with patch.object(GPUDetector, '__init__', lambda x: None):
            detector = GPUDetector()
            detector.logger = Mock()
            detector.gpu_monitor = Mock()
            detector.gpu_monitor.get_all_gpus.return_value = [mock_gpu_0, mock_gpu_1]
            detector.detected_gpus = []
            detector.gpu_available = False
            
            # Mock the initialization
            gpus = detector.detect_gpus()
            
            assert len(gpus) == 2
            assert gpus[0].id == 0
            assert gpus[1].id == 1
            assert detector.gpu_available is True
    
    @patch('src.frameworks_drivers.gpu_detector.pynvml')
    def test_detect_gpus_no_gpus_found(self, mock_pynvml):
        """Test GPU detection when no GPUs are found."""
        # Mock pynvml to indicate GPUs are available but return empty list
        mock_pynvml.nvmlInit = Mock()
        mock_pynvml.nvmlShutdown = Mock()
        mock_pynvml.NVMLError = Exception
        
        # Mock the gpu_monitor.get_all_gpus method to return empty list
        with patch.object(GPUDetector, '__init__', lambda x: None):
            detector = GPUDetector()
            detector.logger = Mock()
            detector.gpu_monitor = Mock()
            detector.gpu_monitor.get_all_gpus.return_value = []
            detector.detected_gpus = []
            detector.gpu_available = False
            
            gpus = detector.detect_gpus()
            
            assert len(gpus) == 0
            assert detector.gpu_available is False
    
    @patch('src.frameworks_drivers.gpu_detector.pynvml')
    def test_is_gpu_available(self, mock_pynvml):
        """Test checking if GPU is available."""
        mock_pynvml.nvmlInit = Mock()
        mock_pynvml.NVMLError = Exception
        
        detector = GPUDetector()
        
        # Initially should be False
        assert detector.is_gpu_available() is False
        
        # Set to True and check
        detector.gpu_available = True
        assert detector.is_gpu_available() is True
    
    @patch('src.frameworks_drivers.gpu_detector.pynvml')
    def test_get_available_gpus(self, mock_pynvml):
        """Test getting available GPUs."""
        mock_pynvml.nvmlInit = Mock()
        mock_pynvml.NVMLError = Exception
        
        mock_gpu = GPU(
            id=0,
            name="Test GPU",
            total_memory=8.0,
            free_memory=6.0,
            used_memory=2.0,
            utilization=25.0,
            assigned_models=[]
        )
        
        detector = GPUDetector()
        detector.detected_gpus = [mock_gpu]
        detector.gpu_available = True
        
        available_gpus = detector.get_available_gpus()
        
        assert len(available_gpus) == 1
        assert available_gpus[0].id == 0
    
    @patch('src.frameworks_drivers.gpu_detector.pynvml')
    def test_get_available_gpus_not_available(self, mock_pynvml):
        """Test getting available GPUs when GPU is not available."""
        mock_pynvml.nvmlInit = Mock()
        mock_pynvml.NVMLError = Exception
        
        mock_gpu = GPU(
            id=0,
            name="Test GPU",
            total_memory=8.0,
            free_memory=6.0,
            used_memory=2.0,
            utilization=25.0,
            assigned_models=[]
        )
        
        detector = GPUDetector()
        detector.detected_gpus = [mock_gpu]
        detector.gpu_available = False # GPU not available
        
        available_gpus = detector.get_available_gpus()
        
        assert len(available_gpus) == 0
    
    @patch('src.frameworks_drivers.gpu_detector.pynvml')
    def test_initialize_gpu_environment_success(self, mock_pynvml):
        """Test successful GPU environment initialization."""
        mock_pynvml.nvmlInit = Mock()
        mock_pynvml.nvmlShutdown = Mock()
        mock_pynvml.NVMLError = Exception
        
        # Mock detection to return GPUs
        with patch.object(GPUDetector, 'detect_gpus') as mock_detect:
            mock_detect.return_value = [
                GPU(id=0, name="GPU 0", total_memory=8.0, free_memory=6.0, 
                    used_memory=2.0, utilization=25.0, assigned_models=[])
            ]
            
            detector = GPUDetector()
            result = detector.initialize_gpu_environment()
            
            assert result is True
            mock_detect.assert_called_once()
    
    @patch('src.frameworks_drivers.gpu_detector.pynvml')
    def test_initialize_gpu_environment_failure(self, mock_pynvml):
        """Test GPU environment initialization failure."""
        mock_pynvml.nvmlInit = Mock()
        mock_pynvml.nvmlShutdown = Mock()
        mock_pynvml.NVMLError = Exception
        
        # Mock detection to return empty list
        with patch.object(GPUDetector, 'detect_gpus') as mock_detect:
            mock_detect.return_value = []  # No GPUs detected
            
            detector = GPUDetector()
            result = detector.initialize_gpu_environment()
            
            assert result is False