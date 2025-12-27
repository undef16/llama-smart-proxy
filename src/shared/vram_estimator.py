"""
Utility for estimating VRAM requirements based on model parameters and quantization.
"""
import math
from typing import Optional
from .gguf_utils import GGUFUtils


class VramEstimator:
    """Class for estimating VRAM requirements."""

    @staticmethod
    def estimate_vram_requirements(
        parameters: int,
        quantization_level: str,
        context_length: int = 4096,
        batch_size: int = 1,
        num_layers: Optional[int] = None,
        hidden_size: Optional[int] = None,
        gqa_factor: float = 1.0,
        cache_type_k: str = 'f16',
        cache_type_v: str = 'f16',
        kv_offload: bool = True,
        repack: bool = True,
        no_host: bool = False,
        activation_overhead_factor: float = 0.25,
        additional_overhead: float = 0.1
    ) -> float:
        """
        More accurate VRAM estimate incorporating llama.cpp params like KV offload and cache quantization.

        Args:
            parameters: Number of model parameters (e.g., 7_000_000_000 for 7B model)
            quantization_level: Quantization level (e.g., 'Q4_K_M', 'Q8_0', 'FP16')
            context_length: Context length to account for KV cache
            batch_size: Batch size for inference
            num_layers: Number of transformer layers (optional, improves accuracy)
            hidden_size: Hidden size of the model (optional, improves accuracy)
            gqa_factor: GQA factor (1.0 for no GQA, <1 for GQA, e.g., 0.125 for Llama 70B)
            cache_type_k: KV cache type for K (e.g., 'q4_0', default 'f16')
            cache_type_v: KV cache type for V (e.g., 'q4_0', default 'f16')
            kv_offload: If True, KV cache is offloaded to CPU (0 VRAM contribution)
            repack: If True, slight reduction in model size (~1-2%)
            no_host: If True, add ~5% extra VRAM for additional GPU buffers
            activation_overhead_factor: Buffer for activations (default 0.25)
            additional_overhead: Additional overhead percentage for safety

        Returns:
            Estimated VRAM requirement in GB
        """
        # Quantization bits for model weights (same as before)
        quantization_bits = {
            'FP16': 16, 'Q8_0': 8, 'Q6_K': 6, 'Q5_K_M': 5, 'Q5_K_S': 5,
            'Q4_K_M': 4, 'Q4_K_S': 4, 'Q3_K_M': 3, 'Q3_K_S': 3, 'Q2_K': 2,
            'FP32': 32, 'GGUF': 16
        }
        bits_per_param = quantization_bits.get(quantization_level.upper(), 16)

        # Model weights in bytes, with minor repack adjustment
        model_size_bytes = (parameters * bits_per_param) / 8
        if repack:
            model_size_bytes *= 0.985  # Assume ~1.5% efficiency gain from repacking

        # Effective bytes per element for KV cache types (accounting for block quantization overhead)
        kv_type_bytes = {
            'f32': 4.0,
            'f16': 2.0,
            'bf16': 2.0,
            'q8_0': 1.0625,  # 8 bits + 16-bit scale per 32 elements: (32*1 + 2)/32
            'q4_0': 0.5625,  # 4 bits + 16-bit scale per 32: (16 + 2)/32 bytes
            'q4_1': 0.625,   # 4 bits + 16-bit scale + 16-bit min per 32: (16 + 4)/32
            'iq4_nl': 0.5625,  # Similar to q4_0
            'q5_0': 0.6875,  # 5 bits + 16-bit scale per 32: (20 + 2)/32
            'q5_1': 0.75     # 5 bits + 16-bit scale + 16-bit min per 32: (20 + 4)/32
        }
        k_bytes = kv_type_bytes.get(cache_type_k.lower(), 2.0)
        v_bytes = kv_type_bytes.get(cache_type_v.lower(), 2.0)

        # KV cache calculation
        if kv_offload:
            kv_cache_bytes = 0  # Offloaded to CPU
        else:
            if num_layers is None or hidden_size is None:
                # Fallback heuristic, scaled by average KV bytes
                base_kv_per_token = 200_000
                if parameters <= 1_000_000:
                    scale = 0.1
                elif parameters <= 100_000_000:
                    scale = math.sqrt(parameters / 7_000_000_000) * 2
                else:
                    scale = math.sqrt(parameters / 7_000_000_000) * 3
                avg_kv_bytes = (k_bytes + v_bytes) / 2  # Average for heuristic
                kv_cache_bytes = context_length * batch_size * base_kv_per_token * max(scale, 0.05) * (avg_kv_bytes / 2.0)  # Normalize to f16 base
            else:
                # Precise: layers * (hidden * gqa) * (k_bytes + v_bytes) per token * batch * context
                kv_per_token = num_layers * hidden_size * gqa_factor * (k_bytes + v_bytes)
                kv_cache_bytes = kv_per_token * batch_size * context_length

        # Activation memory
        activation_bytes = model_size_bytes * activation_overhead_factor

        # Total bytes, with no-host adjustment
        total_bytes = model_size_bytes + kv_cache_bytes + activation_bytes
        if no_host:
            total_bytes *= 1.05  # Conservative +5% for extra GPU buffers

        # To GB with overhead
        estimated_gb = (total_bytes / (1024 ** 3)) * (1 + additional_overhead)
        return estimated_gb

    @staticmethod
    def extract_quantization_from_variant(variant: str) -> str:
        """
        Extract quantization level from model variant name.
        
        Args:
            variant: Model variant name (e.g., 'llama-2-7b-chat.Q4_K_M.gguf')
        
        Returns:
            Quantization level string
        """
        # Common quantization patterns in GGUF filenames
        quant_patterns = [
            'Q8_0', 'Q6_K', 'Q5_K_M', 'Q5_K_S', 'Q4_K_M', 'Q4_K_S', 'Q3_K_M', 'Q3_K_S', 'Q2_K',
            'FP16', 'FP32', 'GGUF'
        ]
        
        variant_upper = variant.upper()
        for pattern in quant_patterns:
            if pattern in variant_upper and pattern != 'GGUF':  # Exclude 'GGUF' from direct matches
                return pattern
        
        # Default to FP16 if no quantization pattern found (excluding GGUF)
        return 'FP16'

    @staticmethod
    def estimate_vram_from_gguf_file(
        gguf_path: str,
        context_length: int = 4096,
        batch_size: int = 1,
        cache_type_k: str = 'f16',
        cache_type_v: str = 'f16',
        kv_offload: bool = True,
        repack: bool = True,
        no_host: bool = False,
        activation_overhead_factor: float = 0.25,
        additional_overhead: float = 0.1
    ) -> Optional[float]:
        """
        Estimate VRAM from a GGUF model file by extracting parameters and quantization info directly.

        Args:
            gguf_path: Path to the GGUF model file
            context_length: Context length for KV cache calculation
            batch_size: Batch size for inference
            cache_type_k: KV cache type for K
            cache_type_v: KV cache type for V
            kv_offload: Whether KV offload is enabled
            repack: Whether weight repacking is enabled
            no_host: Whether to bypass host buffers
            activation_overhead_factor: Buffer for activations
            additional_overhead: Additional overhead percentage

        Returns:
            Estimated VRAM in GB or None if extraction fails
        """
        # Extract parameters from GGUF file
        parameters = GGUFUtils.get_model_parameters_from_gguf(gguf_path)
        if parameters is None:
            return None

        # Extract quantization info from GGUF file
        quantization_info = GGUFUtils.get_quantization_info_from_gguf(gguf_path)
        quantization_level = 'FP16'  # Default
        if quantization_info:
            # Convert quantization info to standard format
            quantization_level = VramEstimator.extract_quantization_from_variant(str(quantization_info))

        # Extract additional architectural parameters
        num_layers = GGUFUtils.get_model_layers_from_gguf(gguf_path)
        hidden_size = GGUFUtils.get_model_hidden_size_from_gguf(gguf_path)
        gqa_factor = GGUFUtils.get_model_gqa_factor_from_gguf(gguf_path) or 1.0

        return VramEstimator.estimate_vram_requirements(
            parameters=parameters,
            quantization_level=quantization_level,
            context_length=context_length,
            batch_size=batch_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            gqa_factor=gqa_factor,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            kv_offload=kv_offload,
            repack=repack,
            no_host=no_host,
            activation_overhead_factor=activation_overhead_factor,
            additional_overhead=additional_overhead
        )

    @staticmethod
    def estimate_vram_from_model_details(
        parameters: Optional[int],
        variant: Optional[str],
        context_length: int = 4096
    ) -> Optional[float]:
        """
        Estimate VRAM from model details (parameters and variant).
        
        Args:
            parameters: Number of model parameters
            variant: Model variant name
            context_length: Context length for KV cache calculation
        
        Returns:
            Estimated VRAM in GB or None if parameters not provided
        """
        if parameters is None:
            return None
        
        quantization_level = 'FP16'  # Default
        if variant:
            quantization_level = VramEstimator.extract_quantization_from_variant(variant)
        
        return VramEstimator.estimate_vram_requirements(parameters, quantization_level, context_length)