"""
GPU detection and initialization logic using nvidia-ml-py (preferred) or pynvml library.
The nvidia-ml-py library is the new official NVIDIA library that replaces the deprecated pynvml package.
Both libraries have the same API, ensuring compatibility while addressing the deprecation warning.
"""
import logging
from typing import List, Optional

# Import the nvidia-ml-py library (new official library) or pynvml (deprecated but compatible)
try:
   import pynvml  # This will import either nvidia-ml-py or pynvml (both use same import)
except ImportError:
   pynvml = None

from src.entities.gpu import GPU
from src.frameworks_drivers.gpu_monitor import GPUMonitor
from src.frameworks_drivers.config import Config


class GPUDetector:
    """Handles GPU detection and initialization logic."""
    
    def __init__(self, config: Optional[Config] = None):
        self.logger = logging.getLogger(__name__)
        self.gpu_monitor = GPUMonitor(config)
        self.detected_gpus: List[GPU] = []
        self.gpu_available = False
        self.pynvml_available = False
    
    def detect_gpus(self) -> List[GPU]:
        """Detect available GPUs and return their information."""
        self.logger.info("Detecting GPUs...")
        
        # Check if pynvml library is available
        if pynvml is None:
            self.logger.warning("pynvml/nvidia-ml-py not available, running in CPU-only mode.")
            self.gpu_available = False
            self.pynvml_available = False
            return []
        
        # Check if GPUs are available
        try:
            pynvml.nvmlInit()
            self.gpu_available = True
            pynvml.nvmlShutdown()
        except (pynvml.NVMLError, Exception):
            self.logger.warning("No NVIDIA GPUs detected or pynvml not available. Running in CPU-only mode.")
            self.gpu_available = False
            return []
        
        # Get GPU information
        self.detected_gpus = self.gpu_monitor.get_all_gpus()
        
        if not self.detected_gpus:
            self.logger.warning("NVIDIA drivers found but no GPUs detected.")
            self.gpu_available = False
            return []
        
        self.logger.info(f"Detected {len(self.detected_gpus)} GPU(s)")
        for gpu in self.detected_gpus:
            self.logger.info(f"  GPU {gpu.id}: {gpu.name} - {gpu.total_memory:.1f}GB memory")
        
        return self.detected_gpus
    
    def is_gpu_available(self) -> bool:
        """Check if GPU hardware is available."""
        return self.gpu_available
    
    def is_pynvml_available(self) -> bool:
        """Check if pynvml library is available."""
        return pynvml is not None
    
    def get_available_gpus(self) -> List[GPU]:
        """Get currently available GPUs."""
        if not self.gpu_available:
            return []
        return self.detected_gpus
    
    def initialize_gpu_environment(self) -> bool:
        """Initialize the GPU environment and return success status."""
        try:
            gpus = self.detect_gpus()
            return len(gpus) > 0
        except Exception as e:
            self.logger.error(f"Error initializing GPU environment: {e}")
            return False