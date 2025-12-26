"""
Tests for VRAM estimation functionality.
"""
import pytest

from src.utils.vram_estimator import VramEstimator


class TestVRAMEstimator:
    """Test cases for VRAM estimation utilities."""
    
    def test_estimate_vram_requirements_basic(self):
        """Test basic VRAM estimation."""
        # Test with 7B parameters, Q4_K_M quantization
        vram = VramEstimator.estimate_vram_requirements(
            parameters=7_000_000,  # 7 billion parameters
            quantization_level='Q4_K_M',
            context_length=4096,
            additional_overhead=0.1  # 10% overhead
        )
        
        # Should be significantly less than FP16 (14GB+) due to quantization
        assert vram > 0
        assert vram < 14.0  # Should be less than FP16 requirement
    
    def test_estimate_vram_requirements_fp16(self):
        """Test VRAM estimation for FP16 model."""
        vram = VramEstimator.estimate_vram_requirements(
            parameters=7_000_000_000,  # 7 billion parameters
            quantization_level='FP16',
            context_length=4096,
            additional_overhead=0.0  # No overhead for this test
        )
        
        # FP16 should require roughly 14GB for 7B model (7B * 2 bytes / 1024^3)
        # Plus KV cache which adds significant overhead
        assert vram > 14.0
    
    def test_estimate_vram_requirements_different_quantizations(self):
        """Test VRAM estimation for different quantization levels."""
        params = 7_000_000_000
        
        vram_q8 = VramEstimator.estimate_vram_requirements(params, 'Q8_0')
        vram_q4 = VramEstimator.estimate_vram_requirements(params, 'Q4_K_M')
        vram_q2 = VramEstimator.estimate_vram_requirements(params, 'Q2_K')
        
        # Lower quantization should require less VRAM
        assert vram_q2 < vram_q4 < vram_q8
    
    def test_estimate_vram_requirements_invalid_quantization(self):
        """Test VRAM estimation with invalid quantization (should default to FP16)."""
        vram = VramEstimator.estimate_vram_requirements(
            parameters=7_000_000,
            quantization_level='INVALID_QUANT',
            context_length=4096,
            additional_overhead=0.0
        )
        
        # Should default to FP16 (16 bits per parameter)
        assert vram > 0
    
    def test_extract_quantization_from_variant(self):
        """Test extracting quantization from model variant names."""
        variants_and_expected = [
            ('model.Q4_K_M.gguf', 'Q4_K_M'),
            ('model.Q8_0.gguf', 'Q8_0'),
            ('model.gguf', 'FP16'),  # Default when no quantization found
            ('llama-2-7b.Q5_K_M.gguf', 'Q5_K_M'),
            ('mistral-7b-instruct-v0.1.Q2_K.gguf', 'Q2_K'),
            ('model.fp16.gguf', 'FP16'),
            ('model.FP32.bin', 'FP32'),
        ]
        
        for variant, expected in variants_and_expected:
            result = VramEstimator.extract_quantization_from_variant(variant)
            assert result == expected, f"Failed for variant: {variant}"
    
    def test_estimate_vram_from_model_details(self):
        """Test estimating VRAM from model details."""
        # With parameters and variant
        vram = VramEstimator.estimate_vram_from_model_details(
            parameters=7_000_000_000,
            variant='model.Q4_K_M.gguf'
        )
        
        assert vram is not None
        assert vram > 0
        
        # With parameters but no variant (should default to FP16)
        vram_default = VramEstimator.estimate_vram_from_model_details(
            parameters=7_000_000_000,
            variant=None
        )
        
        assert vram_default is not None
        assert vram_default > vram  # FP16 should require more VRAM than Q4
        
        # Without parameters (should return None)
        vram_none = VramEstimator.estimate_vram_from_model_details(
            parameters=None,
            variant='model.Q4_K_M.gguf'
        )
        
        assert vram_none is None


def test_vram_estimator_edge_cases():
    """Test edge cases for VRAM estimation."""
    # Test with very small model
    small_vram = VramEstimator.estimate_vram_requirements(
        parameters=1_000_000,  # 1M parameters
        quantization_level='Q4_K_M'
    )
    assert small_vram > 0
    assert small_vram < 1.0  # Should be less than 1GB
    
    # Test with very large model
    large_vram = VramEstimator.estimate_vram_requirements(
        parameters=70_000_000_000,  # 70B parameters
        quantization_level='Q4_K_M'
    )
    assert large_vram > 10  # Should be significantly more than 10GB
    
    # Test different context lengths (disable KV offload to see the difference)
    vram_short = VramEstimator.estimate_vram_requirements(
        parameters=7_000_000_000,
        quantization_level='Q4_K_M',
        context_length=512, # Short context
        kv_offload=False
    )

    vram_long = VramEstimator.estimate_vram_requirements(
        parameters=7_000_000_000,
        quantization_level='Q4_K_M',
        context_length=8192,  # Long context
        kv_offload=False
    )

    # Longer context should require more VRAM due to larger KV cache
    assert vram_long > vram_short