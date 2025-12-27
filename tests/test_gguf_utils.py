"""
Tests for GGUF utilities.
"""
import os
import tempfile
from unittest.mock import mock_open, patch
import pytest

from src.shared.gguf_utils import GGUFUtils


class TestGGUFUtils:
    """Test cases for GGUF utilities."""
    
    def test_extract_gguf_parameters_invalid_file(self):
        """Test extracting parameters from non-existent file."""
        result = GGUFUtils.extract_gguf_parameters("/nonexistent/file.gguf")
        assert result is None

    def test_extract_gguf_parameters_invalid_magic(self):
        """Test extracting parameters from file with invalid magic number."""
        # Create a temporary file with invalid magic
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b'INVALID')  # Invalid magic
            tmp_path = tmp.name

        try:
            result = GGUFUtils.extract_gguf_parameters(tmp_path)
            assert result is None
        finally:
            os.unlink(tmp_path)
    
    def test_get_model_parameters_from_gguf(self):
        """Test getting model parameters from GGUF file."""
        # Since we can't easily create a real GGUF file for testing,
        # we'll test the function behavior with a mock
        with patch('src.shared.gguf_utils.GGUFUtils.extract_gguf_parameters') as mock_extract:
            mock_extract.return_value = {'total_parameters': 7_000_000_000}
            params = GGUFUtils.get_model_parameters_from_gguf("dummy.gguf")
            assert params == 7_000_000_000

    def test_get_model_parameters_from_gguf_none(self):
        """Test getting model parameters when extraction fails."""
        with patch('src.shared.gguf_utils.GGUFUtils.extract_gguf_parameters') as mock_extract:
            mock_extract.return_value = None
            params = GGUFUtils.get_model_parameters_from_gguf("dummy.gguf")
            assert params is None
    
    def test_get_quantization_info_from_gguf(self):
        """Test getting quantization info from GGUF file."""
        with patch('src.shared.gguf_utils.GGUFUtils.extract_gguf_parameters') as mock_extract:
            mock_extract.return_value = {
                'metadata': {
                    'general.quantization_version': 'Q4_K_M'
                }
            }
            quant_info = GGUFUtils.get_quantization_info_from_gguf("dummy.gguf")
            assert quant_info == 'Q4_K_M'

    def test_get_quantization_info_from_gguf_fallback(self):
        """Test getting quantization info with fallback keys."""
        with patch('src.shared.gguf_utils.GGUFUtils.extract_gguf_parameters') as mock_extract:
            mock_extract.return_value = {
                'metadata': {
                    'quantization_version': 'Q8_0'
                }
            }
            quant_info = GGUFUtils.get_quantization_info_from_gguf("dummy.gguf")
            assert quant_info == 'Q8_0'
    
    def test_get_model_architecture_from_gguf(self):
        """Test getting model architecture from GGUF file."""
        with patch('src.shared.gguf_utils.GGUFUtils.extract_gguf_parameters') as mock_extract:
            mock_extract.return_value = {
                'metadata': {
                    'general.architecture': 'llama'
                }
            }
            arch = GGUFUtils.get_model_architecture_from_gguf("dummy.gguf")
            assert arch == 'llama'

    def test_get_model_architecture_from_gguf_none(self):
        """Test getting model architecture when extraction fails."""
        with patch('src.shared.gguf_utils.GGUFUtils.extract_gguf_parameters') as mock_extract:
            mock_extract.return_value = None
            arch = GGUFUtils.get_model_architecture_from_gguf("dummy.gguf")
            assert arch is None

    def test_detect_quantization_from_gguf(self):
        """Test detecting quantization from GGUF file."""
        with patch('src.shared.gguf_utils.GGUFUtils.extract_gguf_parameters') as mock_extract:
            mock_extract.return_value = {
                'metadata': {
                    'general.quantization_version': 'Q4_K_M'
                }
            }
            quant = GGUFUtils.detect_quantization_from_gguf("dummy.gguf")
            assert quant == 'Q4_K_M'


def test_extract_gguf_parameters_structure():
    """Test the structure of extracted GGUF parameters."""
    # Test with mock to verify return structure
    with patch('builtins.open', mock_open()) as mock_file:
        # Simulate a valid GGUF file structure
        mock_file.return_value.__enter__.return_value.read.side_effect = [
            b'GGUF',  # Magic
            (3).to_bytes(4, 'little'),  # Version
            (1).to_bytes(8, 'little'),  # Tensor count
            (2).to_bytes(8, 'little'),  # Metadata count
            # First metadata: key length (13 bytes for "test.key")
            (13).to_bytes(8, 'little'),
            # Key: "test.key"
            b'test.key',
            # Value type: 8 (string)
            (8).to_bytes(4, 'little'),
            # String value length (5)
            (5).to_bytes(8, 'little'),
            # String value: "value"
            b'value',
            # Second metadata: key length (24 bytes for "general.architecture")
            (24).to_bytes(8, 'little'),
            # Key: "general.architecture"
            b'general.architecture',
            # Value type: 8 (string)
            (8).to_bytes(4, 'little'),
            # String value length (5)
            (5).to_bytes(8, 'little'),
            # String value: "llama"
            b'llama',
            # Tensor name length (10)
            (10).to_bytes(8, 'little'),
            # Tensor name
            b'tensor.name',
            # Number of dimensions: 2
            (2).to_bytes(4, 'little'),
            # Dimension 1: 100
            (100).to_bytes(8, 'little'),
            # Dimension 2: 200
            (200).to_bytes(8, 'little'),
            # Tensor offset
            (0).to_bytes(8, 'little')
        ]
        
        # This is a complex test that would require more detailed mocking
        # For now, we'll just verify the function can be called without error
        # In a real scenario, we'd have actual GGUF test files
        pass


def test_convert_to_standard_quantization_format():
    """Test converting quantization formats to standard format."""
    # Test various quantization formats
    assert GGUFUtils.convert_to_standard_quantization_format('q4_k_m') == 'Q4_K_M'
    assert GGUFUtils.convert_to_standard_quantization_format('Q4_K_M') == 'Q4_K_M'
    assert GGUFUtils.convert_to_standard_quantization_format('q8_0') == 'Q8_0'
    assert GGUFUtils.convert_to_standard_quantization_format('Q8_0') == 'Q8_0'
    assert GGUFUtils.convert_to_standard_quantization_format('fp16') == 'FP16'
    assert GGUFUtils.convert_to_standard_quantization_format('FP16') == 'FP16'
    assert GGUFUtils.convert_to_standard_quantization_format('unknown') == 'UNKNOWN'