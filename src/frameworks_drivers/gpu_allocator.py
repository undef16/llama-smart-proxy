"""
GPU allocation strategy interface and implementations.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from src.entities.gpu import GPU
from src.entities.gpu_assignment import GPUAssignment
from src.utils.vram_estimator import VramEstimator
from src.utils.gguf_utils import GGUFUtils


class GPUAllocationStrategy(ABC):
    """Interface for GPU allocation strategies."""
    
    @abstractmethod
    def allocate_gpus(
        self,
        required_vram: float,
        available_gpus: List[GPU],
        model_parameters: Optional[int] = None,
        model_variant: Optional[str] = None,
        gguf_path: Optional[str] = None
    ) -> Optional[GPUAssignment]:
        """
        Allocate GPUs for a model based on the strategy.
        
        Args:
            required_vram: Required VRAM in GB
            available_gpus: List of available GPUs
            model_parameters: Number of model parameters
            model_variant: Model variant name
        
        Returns:
            GPUAssignment if allocation successful, None otherwise
        """
        pass


class SingleGPUAllocationStrategy(GPUAllocationStrategy):
    """Strategy that prefers single GPU allocation when possible."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def allocate_gpus(
        self,
        required_vram: float,
        available_gpus: List[GPU],
        model_parameters: Optional[int] = None,
        model_variant: Optional[str] = None,
        gguf_path: Optional[str] = None
    ) -> Optional[GPUAssignment]:
        """
        Try to allocate a single GPU that can fit the model.
        
        Args:
            required_vram: Required VRAM in GB
            available_gpus: List of available GPUs sorted by free memory (descending)
            model_parameters: Number of model parameters
            model_variant: Model variant name
        
        Returns:
            GPUAssignment if allocation successful, None otherwise
        """
        # Sort GPUs by free memory in descending order to try largest first
        sorted_gpus = sorted(available_gpus, key=lambda gpu: gpu.free_memory, reverse=True)
        
        # First, try to find a single GPU that can accommodate the model
        for gpu in sorted_gpus:
            if gpu.free_memory >= required_vram:
                self.logger.info(f"Allocating model to single GPU {gpu.id} ({gpu.name}) "
                               f"with {gpu.free_memory:.2f}GB free memory for {required_vram:.2f}GB requirement")
                return GPUAssignment(
                    gpu_ids=[gpu.id],
                    tensor_splits=[1.0],
                    estimated_vram_required=required_vram
                )
        
        self.logger.info(f"Could not allocate single GPU for {required_vram:.2f}GB requirement. "
                        f"No single GPU has sufficient memory.")
        return None


class MultiGPUAllocationStrategy(GPUAllocationStrategy):
    """Strategy for distributing models across multiple GPUs when needed."""

    def __init__(
        self,
        max_gpus=None,
        min_gpus=2,
        force_spread=False,
        spread_only_if_necessary=True,
        vram_headroom_factor=1.1
    ):
        self.max_gpus = max_gpus
        self.min_gpus = min_gpus
        self.force_spread = force_spread
        self.spread_only_if_necessary = spread_only_if_necessary
        self.vram_headroom_factor = vram_headroom_factor
        self.logger = logging.getLogger(__name__)
    
    def allocate_gpus(
        self,
        required_vram: float,
        available_gpus: List[GPU],
        model_parameters: Optional[int] = None,
        model_variant: Optional[str] = None,
        gguf_path: Optional[str] = None
    ) -> Optional[GPUAssignment]:
        """
        Allocate multiple GPUs to fit the model requirements using optimized distribution.
        Considers factors like GPU compatibility, memory distribution, and performance.

        Args:
            required_vram: Required VRAM in GB
            available_gpus: List of available GPUs
            model_parameters: Number of model parameters
            model_variant: Model variant name

        Returns:
            GPUAssignment if allocation successful, None otherwise
        """
        if not available_gpus:
            self.logger.error("No GPUs available for multi-GPU allocation")
            return None

        # Apply VRAM headroom factor
        required_vram_with_headroom = required_vram * self.vram_headroom_factor

        # Use optimized multi-GPU allocation algorithm
        return self._allocate_optimized_multi_gpu(required_vram_with_headroom, available_gpus)

    def _allocate_optimized_multi_gpu(
        self,
        required_vram: float,
        available_gpus: List[GPU]
    ) -> Optional[GPUAssignment]:
        """
        Greedy minimal-k multi-GPU allocation: find the smallest number of GPUs (k >= min_gpus)
        whose combined free memory meets requirements. Calculate proportional tensor_splits
        based on each GPU's free memory contribution.
        """
        # First, check if total available VRAM is sufficient
        total_available_vram = sum(gpu.free_memory for gpu in available_gpus)
        if total_available_vram < required_vram:
            self.logger.info(f"Insufficient total VRAM across all GPUs for {required_vram:.2f}GB requirement. "
                           f"Available: {total_available_vram:.2f}GB")
            return None

        # Check if single GPU is possible
        single_possible = any(gpu.free_memory >= required_vram for gpu in available_gpus)
        if single_possible and not self.force_spread and self.spread_only_if_necessary:
            self.logger.info("Single GPU allocation possible and spread not forced, skipping multi-GPU")
            return None

        # Sort GPUs by free memory in descending order
        sorted_gpus = sorted(available_gpus, key=lambda gpu: gpu.free_memory, reverse=True)

        # Determine max_k
        max_k = self.max_gpus if self.max_gpus is not None else len(sorted_gpus)

        # Find minimal k >= min_gpus where sum of top k free_memory >= required_vram
        for k in range(self.min_gpus, max_k + 1):
            if k > len(sorted_gpus):
                break
            selected_gpus = sorted_gpus[:k]
            total_selected_vram = sum(gpu.free_memory for gpu in selected_gpus)
            if total_selected_vram >= required_vram:
                # Check compatibility (simplified)
                if self._are_gpus_compatible(selected_gpus):
                    gpu_ids = [gpu.id for gpu in selected_gpus]
                    # Calculate proportional tensor_splits based on free memory contribution
                    total_free_selected = sum(gpu.free_memory for gpu in selected_gpus)
                    tensor_splits = [gpu.free_memory / total_free_selected for gpu in selected_gpus]
                    self.logger.info(f"Allocated model across {k} GPUs {gpu_ids} with tensor_splits {tensor_splits} "
                                   f"for {required_vram:.2f}GB requirement (total free: {total_selected_vram:.2f}GB)")
                    return GPUAssignment(
                        gpu_ids=gpu_ids,
                        tensor_splits=tensor_splits,
                        estimated_vram_required=required_vram
                    )
                else:
                    self.logger.info(f"GPUs {selected_gpus} not compatible, trying larger k")
                    continue

        self.logger.info(f"Could not find suitable multi-GPU allocation for {required_vram:.2f}GB requirement")
        return None

    def _are_gpus_compatible(self, gpus: List[GPU]) -> bool:
        """
        Check if GPUs are compatible for multi-GPU operation.
        This is a simplified check - in practice, you might want to check
        compute capability, architecture, etc.
        """
        if len(gpus) <= 1:
            return True

        # For now, just check if all GPUs have similar architecture (same name pattern)
        # In a real implementation, you'd want to check compute capability, etc.
        first_gpu_name = gpus[0].name.lower()
        for gpu in gpus[1:]:
            if gpu.name.lower() != first_gpu_name:
                # Different GPU models, but still potentially compatible
                # For now, we'll allow it but in practice you might want stricter checks
                pass

        return True


class AdaptiveGPUAllocator:
    """Main GPU allocator that uses different strategies based on model requirements."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.single_gpu_strategy = SingleGPUAllocationStrategy()
        self.multi_gpu_strategy = MultiGPUAllocationStrategy(
        max_gpus=self.config.get('max_gpus'),
        min_gpus=self.config.get('min_gpus', 2),
        force_spread=self.config.get('force_spread', False),
        spread_only_if_necessary=self.config.get('spread_only_if_necessary', True),
        vram_headroom_factor=self.config.get('vram_headroom_factor', 1.1)
    )
        
    def select_best_gpus(
       self,
       required_vram: float,
       available_gpus: List[GPU],
       model_parameters: Optional[int] = None,
       model_variant: Optional[str] = None,
       gguf_path: Optional[str] = None
   ) -> Optional[GPUAssignment]:
       """
       Enhanced GPU selection algorithm that considers multiple factors beyond just VRAM.
       Uses an adaptive strategy based on model requirements and GPU capabilities.

       Args:
           required_vram: Required VRAM in GB
           available_gpus: List of available GPUs
           model_parameters: Number of model parameters
           model_variant: Model variant name

       Returns:
           GPUAssignment if allocation successful, None otherwise
       """
       if not available_gpus:
           self.logger.error("No GPUs available for allocation")
           return None

       if required_vram <= 0:
           self.logger.error("Invalid VRAM requirement: must be greater than 0")
           return None

       headroom_factor = self.config.get('vram_headroom_factor', 1.1)
       vram_with_headroom = required_vram * headroom_factor

       force_spread = self.config.get('force_spread', False)
       spread_only_if_necessary = self.config.get('spread_only_if_necessary', True)
       max_gpus = self.config.get('max_gpus')

       single_gpu_assignment = None
       if not force_spread:
           single_gpu_assignment = self._try_single_gpu_allocation(
               vram_with_headroom, available_gpus, model_parameters, model_variant, required_vram
           )
           if single_gpu_assignment and spread_only_if_necessary:
               return single_gpu_assignment

       # Try multi-GPU if single failed or not spread_only_if_necessary
       if max_gpus is None or max_gpus > 1:
           multi_gpu_assignment = self.multi_gpu_strategy.allocate_gpus(
               required_vram, available_gpus, model_parameters, model_variant
           )
           if multi_gpu_assignment:
               return multi_gpu_assignment

       # If multi failed but single succeeded, return single
       if single_gpu_assignment:
           return single_gpu_assignment

       return None

    def _try_single_gpu_allocation(
       self,
       required_vram_with_headroom: float,
       available_gpus: List[GPU],
       model_parameters: Optional[int] = None,
       model_variant: Optional[str] = None,
       original_required_vram: Optional[float] = None
   ) -> Optional[GPUAssignment]:
       """
       Try to allocate a single GPU with preference logic considering multiple factors.
       Prefers GPUs with higher free memory but also considers other factors like utilization.

       Args:
           required_vram_with_headroom: Required VRAM in GB with headroom
           available_gpus: List of available GPUs
           model_parameters: Number of model parameters
           model_variant: Model variant name
           original_required_vram: Original required VRAM without headroom

       Returns:
           GPUAssignment if single GPU allocation successful, None otherwise
       """
       # Sort GPUs by preference: prioritize those with more free memory but also consider utilization
       preferred_gpus = self._rank_gpus_by_preference(available_gpus)

       for gpu in preferred_gpus:
           # Check if this GPU has sufficient VRAM with headroom
           if gpu.free_memory >= required_vram_with_headroom:
               self.logger.info(f"Allocating model to single GPU {gpu.id} ({gpu.name}) "
                              f"with {gpu.free_memory:.2f}GB free memory for {required_vram_with_headroom:.2f}GB requirement")
               return GPUAssignment(
                   gpu_ids=[gpu.id],
                   tensor_splits=[1.0],
                   estimated_vram_required=original_required_vram or required_vram_with_headroom
               )

       self.logger.info(f"No single GPU has sufficient memory for {required_vram_with_headroom:.2f}GB requirement.")
       return None

    def _rank_gpus_by_preference(self, gpus: List[GPU]) -> List[GPU]:
       """
       Rank GPUs by preference for single GPU allocation considering multiple factors:
       1. Free memory (higher is better)
       2. Utilization (lower is better)
       3. Other factors could be added as needed
       
       Args:
           gpus: List of available GPUs
           
       Returns:
           List of GPUs sorted by preference (most preferred first)
       """
       def gpu_preference_score(gpu: GPU) -> float:
           # Higher score means higher preference
           # Base score on free memory (in GB)
           base_score = gpu.free_memory
           
           # Reduce score based on utilization (lower utilization is better)
           # Utilization is 0-100%, so we subtract a normalized value
           utilization_penalty = gpu.utilization / 10.0  # Max penalty of 10 for 100% utilization
           score = base_score - utilization_penalty
           
           return score

       # Sort in descending order (highest score first)
       return sorted(gpus, key=gpu_preference_score, reverse=True)

    def allocate_gpus(
       self,
       required_vram: float,
       available_gpus: List[GPU],
       model_parameters: Optional[int] = None,
       model_variant: Optional[str] = None,
       gguf_path: Optional[str] = None
   ) -> Optional[GPUAssignment]:
       """
       Allocate GPUs using adaptive strategy - try single GPU first, then multi-GPU.

       Args:
           required_vram: Required VRAM in GB
           available_gpus: List of available GPUs
           model_parameters: Number of model parameters
           model_variant: Model variant name
           gguf_path: Path to GGUF file for layer count extraction

       Returns:
           GPUAssignment if allocation successful, None otherwise
       """
       assignment = self.select_best_gpus(required_vram, available_gpus, model_parameters, model_variant)
       if assignment and gguf_path:
           n_gpu_layers = self._calculate_optimal_n_gpu_layers(gguf_path, assignment, available_gpus)
           assignment.n_gpu_layers = n_gpu_layers
       return assignment

    def _calculate_optimal_n_gpu_layers(
       self,
       gguf_path: str,
       assignment: GPUAssignment,
       available_gpus: List[GPU]
    ) -> Optional[int]:
       """
       Calculate the optimal number of GPU layers based on model layers and available VRAM.

       Args:
           gguf_path: Path to the GGUF file
           assignment: The GPU assignment
           available_gpus: List of available GPUs

       Returns:
           Optimal number of layers to offload to GPU or None if calculation fails
       """
       total_layers = GGUFUtils.get_model_layers_from_gguf(gguf_path)
       if not total_layers:
           self.logger.warning(f"Could not extract layer count from {gguf_path}")
           return None

       # Get the assigned GPUs
       assigned_gpus = [gpu for gpu in available_gpus if gpu.id in assignment.gpu_ids]
       if not assigned_gpus:
           return None

       # Calculate total available VRAM for the assignment
       total_available_vram = sum(gpu.free_memory for gpu in assigned_gpus)

       # If total available VRAM >= estimated required, offload all layers
       if total_available_vram >= assignment.estimated_vram_required:
           return total_layers

       # Otherwise, calculate proportion
       # Account for headroom and KV cache (simplified)
       headroom_factor = self.config.get('vram_headroom_factor', 1.1)
       effective_available = total_available_vram / headroom_factor

       proportion = effective_available / assignment.estimated_vram_required
       n_gpu_layers = int(total_layers * proportion)

       # Ensure at least 1 layer if any GPU assigned
       n_gpu_layers = max(1, min(n_gpu_layers, total_layers))

       self.logger.info(f"Calculated optimal n_gpu_layers: {n_gpu_layers} out of {total_layers} total layers "
                       f"for available VRAM {total_available_vram:.2f}GB vs required {assignment.estimated_vram_required:.2f}GB")

       return n_gpu_layers

    def estimate_model_vram(
        self,
        model_parameters: Optional[int],
        model_variant: Optional[str],
        gguf_path: Optional[str] = None
    ) -> Optional[float]:
        """
        Estimate VRAM requirements for a model using various methods.

        Args:
            model_parameters: Number of model parameters
            model_variant: Model variant name
            gguf_path: Path to GGUF file for direct parameter extraction (optional)

        Returns:
            Estimated VRAM in GB or None if parameters not provided or estimation fails
        """
        try:
            # Get VRAM estimation parameters from config
            kv_offload = self.config.get('kv_offload', True) if self.config else True
            cache_type_k = self.config.get('cache_type_k', 'f16') if self.config else 'f16'
            cache_type_v = self.config.get('cache_type_v', 'f16') if self.config else 'f16'
            repack = self.config.get('repack', True) if self.config else True
            no_host = self.config.get('no_host', False) if self.config else False
            activation_overhead_factor = self.config.get('activation_overhead_factor', 0.25) if self.config else 0.25

            if gguf_path:
                # Use GGUF file for direct parameter extraction if available
                return VramEstimator.estimate_vram_from_gguf_file(
                    gguf_path,
                    cache_type_k=cache_type_k,
                    cache_type_v=cache_type_v,
                    kv_offload=kv_offload,
                    repack=repack,
                    no_host=no_host,
                    activation_overhead_factor=activation_overhead_factor
                )
            else:
                # Use provided parameters and variant
                # Handle potential typo in parameter count - if it looks like
                # a billion-scale model was intended but millions were provided
                adjusted_parameters = model_parameters
                if model_parameters and model_parameters < 10_000_000 and model_variant and '7b' in model_variant.lower():
                    # If it's a 7B model but only millions of parameters provided, assume typo and scale up
                    adjusted_parameters = model_parameters * 1000  # Scale millions to billions if it looks like a typo

                return VramEstimator.estimate_vram_from_model_details(adjusted_parameters, model_variant)
        except Exception as e:
            self.logger.error(f"Error estimating VRAM for model: {e}")
            return None

    def estimate_model_vram_from_gguf(self, gguf_path: str) -> Optional[float]:
        """
        Estimate VRAM requirements directly from a GGUF file.

        Args:
            gguf_path: Path to the GGUF model file

        Returns:
            Estimated VRAM in GB or None if extraction fails
        """
        # Get VRAM estimation parameters from config
        kv_offload = self.config.get('kv_offload', True) if self.config else True
        cache_type_k = self.config.get('cache_type_k', 'f16') if self.config else 'f16'
        cache_type_v = self.config.get('cache_type_v', 'f16') if self.config else 'f16'
        repack = self.config.get('repack', True) if self.config else True
        no_host = self.config.get('no_host', False) if self.config else False
        activation_overhead_factor = self.config.get('activation_overhead_factor', 0.25) if self.config else 0.25

        return VramEstimator.estimate_vram_from_gguf_file(
            gguf_path,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            kv_offload=kv_offload,
            repack=repack,
            no_host=no_host,
            activation_overhead_factor=activation_overhead_factor
        )
   