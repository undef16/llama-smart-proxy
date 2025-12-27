"""
Shared GPU utilities for safe pynvml import handling.
"""

class GPUUtils:
    """Utility class for GPU-related operations."""

    @staticmethod
    def safe_import_pynvml():
        """
        Safely import pynvml (or nvidia-ml-py) library.
        Returns the module if available, None otherwise.
        """
        try:
            import pynvml  # This will import either nvidia-ml-py or pynvml (both use same import)
            return pynvml
        except ImportError:
            return None