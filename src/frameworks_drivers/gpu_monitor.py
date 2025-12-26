"""
GPU monitoring service using nvidia-ml-py (preferred) or pynvml library to track GPU utilization and memory.
The nvidia-ml-py library is the new official NVIDIA library that replaces the deprecated pynvml package.
Both libraries have the same API, ensuring compatibility while addressing the deprecation warning.
"""
import logging
from typing import List, Optional

# The nvidia-ml-py library uses the same import statement as the old pynvml library (import pynvml)
# When both are installed, Python will import whichever is available, with package resolution determining priority
# nvidia-ml-py is installed first in requirements.txt, so it should take precedence
try:
    import pynvml  # This will import either nvidia-ml-py or pynvml (both use same import)
except ImportError:
    # If neither library is available, we'll handle the error when the library is used
    pynvml = None

from src.entities.gpu import GPU
from src.frameworks_drivers.config import Config


class GPUMonitor:
    """Service for monitoring GPU status and resources using pynvml."""

    def __init__(self, config: Optional[Config] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.initialized = False

        # Check if GPU monitoring is enabled in config
        enable_monitoring = True
        if config and config.gpu:
            enable_monitoring = config.gpu.enable_gpu_monitoring

        if not enable_monitoring:
            self.logger.info("GPU monitoring disabled by configuration")
            return

        # Check if pynvml library is available
        if pynvml is None:
            self.logger.warning("pynvml/nvidia-ml-py not available, GPU monitoring disabled")
            return

        try:
            pynvml.nvmlInit()
            self.initialized = True
            self.logger.info("GPU monitoring initialized successfully")
        except pynvml.NVMLError as e:
            self.logger.warning(f"GPU monitoring not available: {e}")
        except Exception as e:
            self.logger.warning(f"pynvml/nvidia-ml-py not available, GPU monitoring disabled: {e}")
    
    def get_gpu_count(self) -> int:
        """Get the number of available GPUs."""
        if not self.initialized or pynvml is None:
            return 0
        
        try:
            return pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError as e:
            self.logger.error(f"Error getting GPU count: {e}")
            return 0
    
    def get_gpu_info(self, gpu_id: int, assigned_models: Optional[list[str]] = None) -> Optional[GPU]:
        """Get detailed information about a specific GPU."""
        if not self.initialized or pynvml is None:
            return None
        
        if assigned_models is None:
            assigned_models = []
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            
            # Get GPU name
            raw_name = pynvml.nvmlDeviceGetName(handle)
            # Handle different pynvml versions where the return type might be different
            if isinstance(raw_name, bytes):
                name = raw_name.decode('utf-8')
            else:
                name = raw_name
            
            # Get memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Handle different pynvml versions where memory_info attributes might be bytes or numbers
            total_bytes = memory_info.total if isinstance(memory_info.total, int) else int(memory_info.total)
            used_bytes = memory_info.used if isinstance(memory_info.used, int) else int(memory_info.used)
            free_bytes = memory_info.free if isinstance(memory_info.free, int) else int(memory_info.free)
            
            total_memory_gb = total_bytes / (1024**3)
            used_memory_gb = used_bytes / (1024**3)
            free_memory_gb = free_bytes / (1024**3)
            
            # Get utilization
            utilization = 0.0
            try:
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                # Handle different pynvml versions where util_info.gpu might be bytes or number
                utilization = float(util_info.gpu) if isinstance(util_info.gpu, (int, float)) else float(util_info.gpu)
            except pynvml.NVMLError:
                # Some GPUs might not support utilization reporting
                pass
            
            # Get temperature
            temperature = None
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                temperature = float(temp)
            except pynvml.NVMLError:
                # Temperature might not be available
                pass
            
            # Get power usage
            power_usage = None
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                power_usage = float(power)
            except pynvml.NVMLError:
                # Power usage might not be available
                pass
            
            # Get compute capability
            compute_capability = None
            try:
                major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                compute_capability = f"{major}.{minor}"
            except pynvml.NVMLError:
                # Compute capability might not be available
                pass
            
            return GPU(
                id=gpu_id,
                name=name,
                total_memory=total_memory_gb,
                free_memory=free_memory_gb,
                used_memory=used_memory_gb,
                utilization=utilization,
                temperature=temperature,
                power_usage=power_usage,
                assigned_models=assigned_models,
                compute_capability=compute_capability
            )
        except pynvml.NVMLError as e:
            self.logger.error(f"Error getting GPU info for GPU {gpu_id}: {e}")
            return None
    
    def get_all_gpus(self, gpu_assignments: Optional[dict[int, list[str]]] = None) -> List[GPU]:
        """Get information about all available GPUs."""
        if not self.initialized or pynvml is None:
            return []
        
        if gpu_assignments is None:
            gpu_assignments = {}
        
        gpus = []
        gpu_count = self.get_gpu_count()
        
        for i in range(gpu_count):
            # Get the assigned models for this GPU
            assigned_models = gpu_assignments.get(i, [])
            gpu_info = self.get_gpu_info(i, assigned_models)
            if gpu_info:
                gpus.append(gpu_info)
        
        return gpus
    
    def get_gpu_memory_info(self, gpu_id: int) -> Optional[tuple[float, float, float]]:
        """Get memory information for a specific GPU (total, used, free in GB)."""
        if not self.initialized or pynvml is None:
            return None
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Handle different pynvml versions where memory_info attributes might be bytes or numbers
            total_bytes = memory_info.total if isinstance(memory_info.total, int) else int(memory_info.total)
            used_bytes = memory_info.used if isinstance(memory_info.used, int) else int(memory_info.used)
            free_bytes = memory_info.free if isinstance(memory_info.free, int) else int(memory_info.free)
            
            total = total_bytes / (1024**3)
            used = used_bytes / (1024**3)
            free = free_bytes / (1024**3)
            return total, used, free
        except pynvml.NVMLError as e:
            self.logger.error(f"Error getting GPU memory info for GPU {gpu_id}: {e}")
            return None
    
    def get_gpu_utilization(self, gpu_id: int) -> Optional[float]:
        """Get utilization percentage for a specific GPU."""
        if not self.initialized or pynvml is None:
            return None
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util_info.gpu)
        except pynvml.NVMLError as e:
            self.logger.error(f"Error getting GPU utilization for GPU {gpu_id}: {e}")
            return None
    
    def shutdown(self):
        """Clean shutdown of GPU monitoring."""
        if self.initialized and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
                self.logger.info("GPU monitoring shut down successfully")
            except Exception as e:
                self.logger.error(f"Error shutting down GPU monitoring: {e}")
            finally:
                self.initialized = False