"""
TEMPORARY VRAM ACCURACY VALIDATION MODULE

This module provides experimental validation of VRAM estimation accuracy by comparing
estimated VRAM requirements against actual GPU memory usage after model loading.

WARNING: This is a temporary feature for evaluation purposes only.
It will be removed after detailed analysis of estimation accuracy.

Usage:
    validator = VRAMAccuracyValidator()
    baseline = validator.capture_baseline_memory(gpu_ids)
    # ... load model ...
    actual_usage = validator.measure_actual_usage(gpu_ids, baseline)
    accuracy = validator.compare_accuracy(estimated_vram, actual_usage)
    validator.log_accuracy_metrics(model_info, accuracy)
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.entities.gpu import GPU
from src.frameworks_drivers.gpu.gpu_monitor import GPUMonitor
from src.shared.vram_estimator import VramEstimator


@dataclass
class MemorySnapshot:
    """Snapshot of GPU memory state at a point in time."""
    timestamp: datetime
    gpu_memory_used: Dict[int, float]  # GPU ID -> used memory in GB
    total_memory_used: float  # Sum across all GPUs


@dataclass
class VRAMAccuracyResult:
    """Result of VRAM estimation accuracy comparison."""
    model_id: str
    estimated_vram: float
    actual_vram: float
    accuracy_percentage: float  # (estimated / actual) * 100
    variance_gb: float  # estimated - actual
    variance_percentage: float  # ((estimated - actual) / actual) * 100
    gpu_ids: List[int]
    gpu_models: List[str]
    timestamp: datetime
    context_length: int = 4096
    batch_size: int = 1
    quantization_level: Optional[str] = None
    model_parameters: Optional[int] = None


class VRAMAccuracyValidator:
    """
    Temporary validator for checking VRAM estimation accuracy against real usage.

    This class captures baseline GPU memory, measures post-load usage,
    and compares against estimations to validate accuracy.
    """

    def __init__(self, gpu_monitor: Optional[GPUMonitor] = None):
        self.logger = logging.getLogger(__name__)
        self.gpu_monitor = gpu_monitor or GPUMonitor()
        self.accuracy_results: List[VRAMAccuracyResult] = []

        # Accuracy thresholds for logging
        self.accuracy_tolerance = 0.10  # 10% tolerance
        self.significant_variance_threshold = 0.20  # 20% significant variance

    def capture_baseline_memory(self, gpu_ids: List[int]) -> MemorySnapshot:
        """
        Capture baseline GPU memory usage before model loading.

        Args:
            gpu_ids: List of GPU IDs to monitor

        Returns:
            MemorySnapshot with current memory state
        """
        self.logger.info(f"Capturing baseline memory for GPUs: {gpu_ids}")

        gpu_memory_used = {}
        total_used = 0.0

        for gpu_id in gpu_ids:
            memory_info = self.gpu_monitor.get_gpu_memory_info(gpu_id)
            if memory_info:
                total, used, free = memory_info
                gpu_memory_used[gpu_id] = used
                total_used += used
                self.logger.debug(f"GPU {gpu_id} baseline: {used:.2f}GB used")
            else:
                self.logger.warning(f"Could not get memory info for GPU {gpu_id}")
                gpu_memory_used[gpu_id] = 0.0

        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_memory_used=gpu_memory_used,
            total_memory_used=total_used
        )

        self.logger.info(f"Baseline memory captured: {total_used:.2f}GB total across {len(gpu_ids)} GPUs")
        return snapshot

    def measure_actual_usage(self, gpu_ids: List[int], baseline: MemorySnapshot) -> MemorySnapshot:
        """
        Measure actual GPU memory usage after model loading.

        Args:
            gpu_ids: List of GPU IDs that were assigned
            baseline: Baseline memory snapshot taken before loading

        Returns:
            MemorySnapshot with post-load memory state
        """
        self.logger.info(f"Measuring post-load memory for GPUs: {gpu_ids}")

        # Allow some time for memory usage to stabilize
        time.sleep(2.0)

        gpu_memory_used = {}
        total_used = 0.0

        for gpu_id in gpu_ids:
            memory_info = self.gpu_monitor.get_gpu_memory_info(gpu_id)
            if memory_info:
                total, used, free = memory_info
                gpu_memory_used[gpu_id] = used
                total_used += used
                self.logger.debug(f"GPU {gpu_id} post-load: {used:.2f}GB used")
            else:
                self.logger.warning(f"Could not get memory info for GPU {gpu_id}")
                gpu_memory_used[gpu_id] = baseline.gpu_memory_used.get(gpu_id, 0.0)

        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_memory_used=gpu_memory_used,
            total_memory_used=total_used
        )

        # Calculate delta from baseline
        delta = total_used - baseline.total_memory_used
        self.logger.info(f"Post-load memory: {total_used:.2f}GB total, "
                        f"delta from baseline: {delta:.2f}GB")

        return snapshot

    def calculate_actual_vram_used(self, baseline: MemorySnapshot, post_load: MemorySnapshot) -> float:
        """
        Calculate the actual VRAM used by the model (delta from baseline).

        Args:
            baseline: Memory snapshot before loading
            post_load: Memory snapshot after loading

        Returns:
            Actual VRAM used in GB
        """
        # Use the maximum delta across GPUs to account for multi-GPU scenarios
        max_delta = 0.0
        for gpu_id in baseline.gpu_memory_used:
            baseline_used = baseline.gpu_memory_used[gpu_id]
            post_used = post_load.gpu_memory_used.get(gpu_id, baseline_used)
            delta = post_used - baseline_used
            max_delta = max(max_delta, delta)

        actual_used = max(0.0, max_delta)  # Ensure non-negative
        self.logger.info(f"Calculated actual VRAM used: {actual_used:.2f}GB (max delta across GPUs)")
        return actual_used

    def compare_accuracy(self, estimated_vram: float, actual_vram: float) -> Optional[VRAMAccuracyResult]:
        """
        Compare estimated vs actual VRAM usage and calculate accuracy metrics.

        Args:
            estimated_vram: Estimated VRAM requirement in GB
            actual_vram: Actual measured VRAM usage in GB

        Returns:
            VRAMAccuracyResult with comparison metrics, or None if calculation fails
        """
        if actual_vram <= 0:
            self.logger.warning("Actual VRAM usage is zero or negative, cannot calculate accuracy")
            return None

        accuracy_percentage = (estimated_vram / actual_vram) * 100
        variance_gb = estimated_vram - actual_vram
        variance_percentage = (variance_gb / actual_vram) * 100

        result = VRAMAccuracyResult(
            model_id="",  # To be filled by caller
            estimated_vram=estimated_vram,
            actual_vram=actual_vram,
            accuracy_percentage=accuracy_percentage,
            variance_gb=variance_gb,
            variance_percentage=variance_percentage,
            gpu_ids=[],  # To be filled by caller
            gpu_models=[],  # To be filled by caller
            timestamp=datetime.now()
        )

        self.logger.info(f"VRAM Accuracy: {accuracy_percentage:.1f}% "
                        f"(estimated: {estimated_vram:.2f}GB, actual: {actual_vram:.2f}GB, "
                        f"variance: {variance_percentage:+.1f}%)")

        return result

    def log_accuracy_metrics(self, result: VRAMAccuracyResult) -> None:
        """
        Log accuracy metrics and flag significant discrepancies.

        Args:
            result: Accuracy comparison result
        """
        self.accuracy_results.append(result)

        # Log based on accuracy level
        abs_variance_pct = abs(result.variance_percentage)

        if abs_variance_pct <= self.accuracy_tolerance * 100:
            self.logger.info(f"VRAM estimation accurate within {self.accuracy_tolerance*100:.0f}% tolerance")
        elif abs_variance_pct <= self.significant_variance_threshold * 100:
            self.logger.warning(f"VRAM estimation variance: {result.variance_percentage:+.1f}% "
                              f"(estimated: {result.estimated_vram:.2f}GB, actual: {result.actual_vram:.2f}GB)")
        else:
            self.logger.error(f"SIGNIFICANT VRAM estimation discrepancy: {result.variance_percentage:+.1f}% "
                            f"(estimated: {result.estimated_vram:.2f}GB, actual: {result.actual_vram:.2f}GB)")

        # Store result for analysis
        self._store_accuracy_result(result)

    def _store_accuracy_result(self, result: VRAMAccuracyResult) -> None:
        """
        Store accuracy result for later analysis.
        In a real implementation, this would write to a database or file.
        """
        # For now, just keep in memory
        # TODO: Implement persistent storage if needed for analysis
        pass

    def get_accuracy_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of all accuracy measurements.

        Returns:
            Dictionary with accuracy statistics
        """
        if not self.accuracy_results:
            return {"total_measurements": 0}

        accuracies = [r.accuracy_percentage for r in self.accuracy_results]
        variances = [abs(r.variance_percentage) for r in self.accuracy_results]

        return {
            "total_measurements": len(self.accuracy_results),
            "average_accuracy": sum(accuracies) / len(accuracies),
            "median_accuracy": sorted(accuracies)[len(accuracies) // 2],
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "average_variance": sum(variances) / len(variances),
            "within_10_percent": sum(1 for v in variances if v <= 10.0),
            "within_20_percent": sum(1 for v in variances if v <= 20.0),
        }

    def validate_model_loading(self, model_id: str, gpu_ids: List[int],
                             estimated_vram: float, baseline: MemorySnapshot) -> Optional[VRAMAccuracyResult]:
        """
        Complete validation workflow: measure actual usage and compare with estimate.

        Args:
            model_id: Identifier for the model being loaded
            gpu_ids: GPU IDs assigned to the model
            estimated_vram: Estimated VRAM requirement
            baseline: Baseline memory snapshot

        Returns:
            VRAMAccuracyResult if validation successful, None otherwise
        """
        try:
            # Measure actual usage
            post_load = self.measure_actual_usage(gpu_ids, baseline)
            actual_vram = self.calculate_actual_vram_used(baseline, post_load)

            if actual_vram <= 0:
                self.logger.warning(f"No measurable VRAM usage for model {model_id}")
                return None

            # Compare accuracy
            result = self.compare_accuracy(estimated_vram, actual_vram)
    
            if result is None:
                return None
    
            # Fill in model details
            result.model_id = model_id
            result.gpu_ids = gpu_ids

            # Get GPU model names
            gpu_models = []
            for gpu_id in gpu_ids:
                gpu_info = self.gpu_monitor.get_gpu_info(gpu_id)
                if gpu_info:
                    gpu_models.append(gpu_info.name)
            result.gpu_models = gpu_models

            # Log metrics
            self.log_accuracy_metrics(result)

            return result

        except Exception as e:
            self.logger.error(f"Error during VRAM validation for model {model_id}: {e}")
            return None