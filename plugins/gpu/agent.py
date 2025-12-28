"""
GPU VRAM Accuracy Plugin Agent

This agent measures and reports the difference between estimated and actual VRAM usage
for the model being used in chat completions.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from plugins.gpu.vram_accuracy_validator import VRAMAccuracyValidator, MemorySnapshot
from src.shared.vram_estimator import VramEstimator
from src.shared.gguf_utils import GGUFUtils
from src.frameworks_drivers.gpu.gpu_monitor import GPUMonitor


class GPUAgent:
    """
    Agent that reports VRAM accuracy on chat completion responses.
    Activated by /gp command.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validator = VRAMAccuracyValidator()
        self.gpu_monitor = GPUMonitor()
        self._current_model: Optional[str] = None
        self._baseline: Optional[MemorySnapshot] = None

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the incoming request.
        Store model information and capture baseline VRAM usage.
        """
        # Store model info for use in response processing
        self._current_model = request.get("model", "")

        # Capture baseline VRAM usage before processing
        try:
            gpu_ids = self._get_available_gpu_ids()
            if gpu_ids:
                self._baseline = self.validator.capture_baseline_memory(gpu_ids)
                self.logger.debug(f"Captured baseline VRAM: {self._baseline.total_memory_used:.2f}GB")
        except Exception as e:
            self.logger.error(f"Failed to capture baseline VRAM: {e}")
            self._baseline = None

        return request

    def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the response and add VRAM accuracy information.
        """
        try:
            model = self._current_model
            if not model:
                self.logger.warning("No model information available for VRAM accuracy check")
                return response

            # Estimate VRAM requirements
            estimated_vram = self._estimate_vram_for_model(model)
            if estimated_vram is None:
                self.logger.warning(f"Could not estimate VRAM for model {model}")
                return response

            # Measure actual VRAM used during request processing
            actual_vram = self._calculate_actual_vram_used()
            if actual_vram is None:
                self.logger.warning("Could not calculate actual VRAM used")
                return response

            # Calculate accuracy
            accuracy_info = self._calculate_accuracy_info(estimated_vram, actual_vram, model)

            # Add to response
            if "vram_accuracy" not in response:
                response["vram_accuracy"] = accuracy_info

            self.logger.info(f"VRAM Accuracy for {model}: {accuracy_info}")

        except Exception as e:
            self.logger.error(f"Error in GP agent response processing: {e}")

        return response

    def _calculate_actual_vram_used(self) -> Optional[float]:
        """
        Calculate actual VRAM used during request processing (delta from baseline).
        """
        if not self._baseline:
            self.logger.warning("No baseline captured, falling back to current usage")
            return self._measure_current_gpu_usage()

        try:
            gpu_ids = list(self._baseline.gpu_memory_used.keys())
            post_load = self.validator.measure_actual_usage(gpu_ids, self._baseline)
            actual_used = self.validator.calculate_actual_vram_used(self._baseline, post_load)

            if actual_used > 0:
                self.logger.info(f"Actual VRAM used during request: {actual_used:.2f}GB")
                return actual_used
            else:
                self.logger.warning("No measurable VRAM usage during request, falling back to current usage")
                return self._measure_current_gpu_usage()

        except Exception as e:
            self.logger.error(f"Error calculating VRAM delta: {e}")
            return self._measure_current_gpu_usage()

    def _estimate_vram_for_model(self, model: str) -> Optional[float]:
        """
        Estimate VRAM requirements for the given model.
        """
        # Extract parameters from model name
        parameters = GGUFUtils.extract_parameters_from_model_name(model)
        if parameters is None:
            self.logger.warning(f"Could not extract parameters from model name: {model}")
            return None

        # Extract quantization from model name
        quantization = VramEstimator.extract_quantization_from_variant(model)

        # Extract quantization
        quantization_level = VramEstimator.extract_quantization_from_variant(model)

        # Estimate VRAM - assume KV cache is not offloaded (typical for llama.cpp)
        estimated = VramEstimator.estimate_vram_requirements(
            parameters=parameters,
            quantization_level=quantization_level,
            kv_offload=False  # KV cache typically stays on GPU
        )

        return estimated

    def _measure_current_gpu_usage(self) -> Optional[float]:
        """
        Measure current GPU memory usage for the current process (server) across all GPUs.
        Uses NVML per-process APIs similar to the provided example.
        """
        try:
            import os
            current_pid = os.getpid()
            self.logger.info(f"Measuring VRAM usage for process PID: {current_pid}")

            # Import pynvml
            try:
                import pynvml as nvml
            except ImportError:
                self.logger.error("pynvml not available for per-process VRAM measurement")
                return self._fallback_total_gpu_usage()

            # Initialize NVML
            nvml.nvmlInit()

            try:
                device_count = nvml.nvmlDeviceGetCount()
                per_pid_bytes = {}

                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)

                    # Get compute and graphics processes using try_call approach
                    compute_procs = self._try_call(nvml.nvmlDeviceGetComputeRunningProcesses_v3, handle) or \
                                   self._try_call(nvml.nvmlDeviceGetComputeRunningProcesses_v2, handle) or \
                                   self._try_call(nvml.nvmlDeviceGetComputeRunningProcesses, handle) or []

                    graphics_procs = self._try_call(nvml.nvmlDeviceGetGraphicsRunningProcesses_v3, handle) or \
                                    self._try_call(nvml.nvmlDeviceGetGraphicsRunningProcesses_v2, handle) or \
                                    self._try_call(nvml.nvmlDeviceGetGraphicsRunningProcesses, handle) or []

                    # Merge by PID
                    for proc in compute_procs + graphics_procs:
                        used = getattr(proc, "usedGpuMemory", None)
                        if used is not None:
                            per_pid_bytes[proc.pid] = per_pid_bytes.get(proc.pid, 0) + used

                # Find our process
                if current_pid in per_pid_bytes:
                    used_bytes = per_pid_bytes[current_pid]
                    used_gb = used_bytes / (1024 ** 3)
                    self.logger.info(f"Process {current_pid} VRAM usage: {used_gb:.2f}GB")
                    return used_gb
                else:
                    self.logger.warning(f"Process {current_pid} not found in NVML process list, falling back to total GPU usage")
                    self.logger.debug(f"Found PIDs: {list(per_pid_bytes.keys())}")
                    return self._fallback_total_gpu_usage()

            finally:
                nvml.nvmlShutdown()

        except Exception as e:
            self.logger.error(f"Error measuring per-process GPU usage: {e}")
            return self._fallback_total_gpu_usage()

    def _try_call(self, fn, handle):
        """Call an NVML function, return [] if not supported."""
        try:
            return fn(handle)
        except (Exception):  # NVMLError_NotSupported, etc.
            return []

    def _get_available_gpu_ids(self) -> List[int]:
        """Get list of available GPU IDs."""
        gpu_ids = []
        for i in range(10):  # Check first 10 GPUs
            try:
                info = self.gpu_monitor.get_gpu_info(i)
                if info:
                    gpu_ids.append(i)
            except:
                break
        return gpu_ids

    def _get_process_list_nvml(self, nvml, handle, proc_type: str):
        """
        Get process list for a GPU handle using NVML, trying different versions.
        """
        try:
            if proc_type == "compute":
                # Try v3, v2, v1
                for fn_name in ["nvmlDeviceGetComputeRunningProcesses_v3",
                               "nvmlDeviceGetComputeRunningProcesses_v2",
                               "nvmlDeviceGetComputeRunningProcesses"]:
                    try:
                        fn = getattr(nvml, fn_name, None)
                        if fn:
                            return fn(handle)
                    except:
                        continue
            elif proc_type == "graphics":
                # Try v3, v2, v1
                for fn_name in ["nvmlDeviceGetGraphicsRunningProcesses_v3",
                               "nvmlDeviceGetGraphicsRunningProcesses_v2",
                               "nvmlDeviceGetGraphicsRunningProcesses"]:
                    try:
                        fn = getattr(nvml, fn_name, None)
                        if fn:
                            return fn(handle)
                    except:
                        continue
        except:
            pass
        return []

    def _fallback_total_gpu_usage(self) -> Optional[float]:
        """
        Fallback: Measure total GPU memory usage across all GPUs.
        """
        try:
            self.logger.info("Using fallback: measuring total GPU memory usage")

            # Get all available GPUs
            gpu_ids = []
            for i in range(10):  # Check first 10 GPUs
                try:
                    info = self.gpu_monitor.get_gpu_info(i)
                    if info:
                        gpu_ids.append(i)
                except:
                    break

            if not gpu_ids:
                return None

            # Measure memory usage
            total_used = 0.0
            for gpu_id in gpu_ids:
                memory_info = self.gpu_monitor.get_gpu_memory_info(gpu_id)
                if memory_info:
                    total, used, free = memory_info
                    total_used += used

            return total_used

        except Exception as e:
            self.logger.error(f"Error in fallback GPU usage measurement: {e}")
            return None

    def _calculate_accuracy_info(self, estimated: float, actual: float, model: str) -> Dict[str, Any]:
        """
        Calculate VRAM accuracy information.
        """
        accuracy_percentage = (estimated / actual) * 100 if actual > 0 else 0
        variance_gb = estimated - actual
        variance_percentage = (variance_gb / actual) * 100 if actual > 0 else 0

        return {
            "model": model,
            "estimated_vram_gb": round(estimated, 2),
            "actual_vram_gb": round(actual, 2),
            "accuracy_percentage": round(accuracy_percentage, 1),
            "variance_gb": round(variance_gb, 2),
            "variance_percentage": round(variance_percentage, 1),
            "timestamp": datetime.now().isoformat()
        }