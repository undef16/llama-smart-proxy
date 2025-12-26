"""
Utility for extracting model parameters from GGUF files.
"""
from typing import Dict, Any, Optional
import struct
import os


class GGUFUtils:
    @staticmethod
    def extract_gguf_parameters(model_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract model parameters from a GGUF file.
        
        Args:
            model_path: Path to the GGUF model file
            
        Returns:
            Dictionary containing model parameters, quantization info, and other metadata
            or None if extraction fails
        """
        if not os.path.exists(model_path):
            return None
        
        try:
            with open(model_path, 'rb') as f:
                # Read GGUF header
                magic = f.read(4)
                if magic != b'GGUF':
                    return None
                
                # Read version (uint32)
                version_data = f.read(4)
                version = struct.unpack('<I', version_data)[0]
                
                # Read number of tensors (uint64)
                tensor_count_data = f.read(8)
                tensor_count = struct.unpack('<Q', tensor_count_data)[0]
                
                # Read number of metadata key-value pairs (uint64)
                metadata_kv_count_data = f.read(8)
                metadata_kv_count = struct.unpack('<Q', metadata_kv_count_data)[0]
                
                # Extract metadata
                metadata = {}
                for _ in range(metadata_kv_count):
                    # Read key length
                    key_len_data = f.read(8)
                    key_len = struct.unpack('<Q', key_len_data)[0]
                    
                    # Read key
                    key = f.read(key_len).decode('utf-8')
                    
                    # Read value type (uint32)
                    value_type_data = f.read(4)
                    value_type = struct.unpack('<I', value_type_data)[0]
                    
                    # Parse value based on type
                    value = GGUFUtils._parse_metadata_value(f, value_type)
                    metadata[key] = value
                
                # Calculate total parameters from tensor info
                total_parameters = 0
                tensor_info = []
                
                for _ in range(tensor_count):
                    # Read tensor name length
                    name_len_data = f.read(8)
                    name_len = struct.unpack('<Q', name_len_data)[0]
                    
                    # Skip tensor name
                    f.read(name_len)
                    
                    # Read number of dimensions
                    n_dims_data = f.read(4)
                    n_dims = struct.unpack('<I', n_dims_data)[0]
                    
                    # Read dimensions
                    dims = []
                    for i in range(n_dims):
                        dim_data = f.read(8)
                        dim = struct.unpack('<Q', dim_data)[0]
                        dims.append(dim)
                    
                    # Read tensor offset
                    offset_data = f.read(8)
                    # Skip offset data
                    
                    # Calculate parameters in this tensor
                    tensor_params = 1
                    for dim in dims:
                        tensor_params *= dim
                    
                    total_parameters += tensor_params
                    tensor_info.append({
                        'dimensions': dims,
                        'parameters': tensor_params
                    })
                
                return {
                    'version': version,
                    'tensor_count': tensor_count,
                    'metadata': metadata,
                    'total_parameters': total_parameters,
                    'tensor_info': tensor_info
                }
        
        except Exception:
            # If direct parsing fails, return None
            return None

    @staticmethod
    def _parse_metadata_value(f, value_type: int) -> Any:
        """
        Parse metadata value based on its type.
        
        Args:
            f: File object
            value_type: GGUF value type
            
        Returns:
            Parsed value
        """
        # GGUF value types:
        # 0 = UINT8, 1 = INT8, 2 = UINT16, 3 = INT16, 4 = UINT32, 5 = INT32
        # 6 = FLOAT32, 7 = BOOL, 8 = STRING, 9 = ARRAY, 10 = UINT64, 11 = INT64, 12 = FLOAT64
        
        if value_type == 0:  # UINT8
            data = f.read(1)
            return struct.unpack('<B', data)[0]
        elif value_type == 1:  # INT8
            data = f.read(1)
            return struct.unpack('<b', data)[0]
        elif value_type == 2:  # UINT16
            data = f.read(2)
            return struct.unpack('<H', data)[0]
        elif value_type == 3:  # INT16
            data = f.read(2)
            return struct.unpack('<h', data)[0]
        elif value_type == 4:  # UINT32
            data = f.read(4)
            return struct.unpack('<I', data)[0]
        elif value_type == 5:  # INT32
            data = f.read(4)
            return struct.unpack('<i', data)[0]
        elif value_type == 6:  # FLOAT32
            data = f.read(4)
            return struct.unpack('<f', data)[0]
        elif value_type == 7:  # BOOL
            data = f.read(1)
            return bool(struct.unpack('<B', data)[0])
        elif value_type == 8:  # STRING
            str_len_data = f.read(8)
            str_len = struct.unpack('<Q', str_len_data)[0]
            str_data = f.read(str_len)
            return str_data.decode('utf-8')
        elif value_type == 9:  # ARRAY
            array_type_data = f.read(4)
            array_type = struct.unpack('<I', array_type_data)[0]
            array_len_data = f.read(8)
            array_len = struct.unpack('<Q', array_len_data)[0]
            
            array_values = []
            for _ in range(array_len):
                array_values.append(GGUFUtils._parse_metadata_value(f, array_type))
            
            return array_values
        elif value_type == 10:  # UINT64
            data = f.read(8)
            return struct.unpack('<Q', data)[0]
        elif value_type == 11:  # INT64
            data = f.read(8)
            return struct.unpack('<q', data)[0]
        elif value_type == 12:  # FLOAT64
            data = f.read(8)
            return struct.unpack('<d', data)[0]
        else:
            # Unknown type, skip
            return None

    @staticmethod
    def get_model_parameters_from_gguf(model_path: str) -> Optional[int]:
        """
        Extract the total number of parameters from a GGUF file.
        
        Args:
            model_path: Path to the GGUF model file
            
        Returns:
            Total number of parameters or None if extraction fails
        """
        result = GGUFUtils.extract_gguf_parameters(model_path)
        if result:
            return result.get('total_parameters')
        return None

    @staticmethod
    def get_quantization_info_from_gguf(model_path: str) -> Optional[str]:
        """
        Extract quantization information from a GGUF file.
        
        Args:
            model_path: Path to the GGUF model file
            
        Returns:
            Quantization format string or None if extraction fails
        """
        result = GGUFUtils.extract_gguf_parameters(model_path)
        if result:
            metadata = result.get('metadata', {})
            # Common GGUF metadata keys for quantization
            quant_keys = [
                'general.quantization_version',
                'quantization_version',
                'gguf.quantization_version'
            ]
            
            for key in quant_keys:
                if key in metadata:
                    return str(metadata[key])
            
            # Look for architecture-specific quantization info
            arch_key = 'general.architecture'
            if arch_key in metadata:
                arch = metadata[arch_key]
                # Check for architecture-specific quantization parameters
                for meta_key, meta_value in metadata.items():
                    if 'quant' in meta_key.lower():
                        return str(meta_value)
        
        return None

    @staticmethod
    def get_model_architecture_from_gguf(model_path: str) -> Optional[str]:
        """
        Extract model architecture from a GGUF file.

        Args:
            model_path: Path to the GGUF model file

        Returns:
            Model architecture string or None if extraction fails
        """
        result = GGUFUtils.extract_gguf_parameters(model_path)
        if result:
            metadata = result.get('metadata', {})
            return metadata.get('general.architecture')
        return None

    @staticmethod
    def get_model_layers_from_gguf(model_path: str) -> Optional[int]:
        """
        Extract the number of layers from a GGUF file by parsing tensor names.

        Args:
            model_path: Path to the GGUF model file

        Returns:
            Number of layers or None if extraction fails
        """
        if not os.path.exists(model_path):
            return None

        try:
            with open(model_path, 'rb') as f:
                # Read GGUF header
                magic = f.read(4)
                if magic != b'GGUF':
                    return None

                # Read version (uint32)
                version_data = f.read(4)
                version = struct.unpack('<I', version_data)[0]

                # Read number of tensors (uint64)
                tensor_count_data = f.read(8)
                tensor_count = struct.unpack('<Q', tensor_count_data)[0]

                # Read number of metadata key-value pairs (uint64)
                metadata_kv_count_data = f.read(8)
                metadata_kv_count = struct.unpack('<Q', metadata_kv_count_data)[0]

                # Skip metadata for now
                for _ in range(metadata_kv_count):
                    # Read key length
                    key_len_data = f.read(8)
                    key_len = struct.unpack('<Q', key_len_data)[0]

                    # Read key
                    f.read(key_len)

                    # Read value type (uint32)
                    value_type_data = f.read(4)
                    value_type = struct.unpack('<I', value_type_data)[0]

                    # Skip value based on type
                    GGUFUtils._skip_metadata_value(f, value_type)

                # Parse tensor names to find layers
                max_layer = -1
                for _ in range(tensor_count):
                    # Read tensor name length
                    name_len_data = f.read(8)
                    name_len = struct.unpack('<Q', name_len_data)[0]

                    # Read tensor name
                    name = f.read(name_len).decode('utf-8')

                    # Check if it's a block tensor (e.g., blk.0.attn_q.weight)
                    if name.startswith('blk.'):
                        try:
                            # Extract layer number
                            layer_num = int(name.split('.')[1])
                            max_layer = max(max_layer, layer_num)
                        except (IndexError, ValueError):
                            continue

                    # Skip the rest of tensor info
                    # n_dims
                    n_dims_data = f.read(4)
                    n_dims = struct.unpack('<I', n_dims_data)[0]

                    # dimensions
                    for i in range(n_dims):
                        f.read(8)  # dim

                    # offset
                    f.read(8)

                # Layers are 0-indexed, so number of layers is max_layer + 1
                if max_layer >= 0:
                    return max_layer + 1
                else:
                    return None

        except Exception:
            return None

    @staticmethod
    def _skip_metadata_value(f, value_type: int) -> None:
        """
        Skip metadata value based on its type.

        Args:
            f: File object
            value_type: GGUF value type
        """
        if value_type == 0:  # UINT8
            f.read(1)
        elif value_type == 1:  # INT8
            f.read(1)
        elif value_type == 2:  # UINT16
            f.read(2)
        elif value_type == 3:  # INT16
            f.read(2)
        elif value_type == 4:  # UINT32
            f.read(4)
        elif value_type == 5:  # INT32
            f.read(4)
        elif value_type == 6:  # FLOAT32
            f.read(4)
        elif value_type == 7:  # BOOL
            f.read(1)
        elif value_type == 8:  # STRING
            str_len_data = f.read(8)
            str_len = struct.unpack('<Q', str_len_data)[0]
            f.read(str_len)
        elif value_type == 9:  # ARRAY
            array_type_data = f.read(4)
            array_type = struct.unpack('<I', array_type_data)[0]
            array_len_data = f.read(8)
            array_len = struct.unpack('<Q', array_len_data)[0]

            for _ in range(array_len):
                GGUFUtils._skip_metadata_value(f, array_type)
        elif value_type == 10:  # UINT64
            f.read(8)
        elif value_type == 11:  # INT64
            f.read(8)
        elif value_type == 12:  # FLOAT64
            f.read(8)
        else:
            # Unknown type, can't skip properly
            pass

    @staticmethod
    def detect_quantization_from_gguf(model_path: str) -> Optional[str]:
        """
        Detect the specific quantization level from a GGUF file, converting it to
        standard format used for VRAM calculation.
        
        Args:
            model_path: Path to the GGUF model file
            
        Returns:
            Standard quantization level (e.g., 'Q4_K_M', 'Q8_0', etc.) or None if detection fails
        """
        result = GGUFUtils.extract_gguf_parameters(model_path)
        if not result:
            return None
        
        metadata = result.get('metadata', {})
        
        # First, try to get quantization info from metadata
        quant_info = GGUFUtils.get_quantization_info_from_gguf(model_path)
        if quant_info:
            # Convert to standard format
            standard_quant = GGUFUtils.convert_to_standard_quantization_format(quant_info)
            if standard_quant:
                return standard_quant
        
        # If not found in metadata, try to infer from tensor info
        tensor_info = result.get('tensor_info', [])
        if tensor_info:
            # Look for common quantization patterns in tensor names or properties
            # This is a simplified approach - in real GGUF files, the quantization
            # information is typically in the metadata
            for tensor in tensor_info:
                # For now, we'll just return what we can find in metadata
                pass
        
        return None

    @staticmethod
    def convert_to_standard_quantization_format(quant_info: str) -> Optional[str]:
        """
        Convert quantization info to standard format used in VRAM calculation.

        Args:
            quant_info: Raw quantization info from GGUF file

        Returns:
            Standard quantization level or None if conversion fails
        """
        # Common quantization patterns found in GGUF files
        quant_patterns = {
            'q4_k_m': 'Q4_K_M',
            'q4_k_s': 'Q4_K_S',
            'q5_k_m': 'Q5_K_M',
            'q5_k_s': 'Q5_K_S',
            'q6_k': 'Q6_K',
            'q8_0': 'Q8_0',
            'q2_k': 'Q2_K',
            'q3_k_m': 'Q3_K_M',
            'q3_k_s': 'Q3_K_S',
            'fp16': 'FP16',
            'fp32': 'FP32',
        }

        quant_lower = quant_info.lower()
        for pattern, standard in quant_patterns.items():
            if pattern in quant_lower:
                return standard

        # If no pattern matches, return the original as uppercase
        return quant_info.upper()

    @staticmethod
    def get_model_hidden_size_from_gguf(model_path: str) -> Optional[int]:
        """
        Extract the hidden size from a GGUF file.

        Args:
            model_path: Path to the GGUF model file

        Returns:
            Hidden size or None if extraction fails
        """
        result = GGUFUtils.extract_gguf_parameters(model_path)
        if result:
            metadata = result.get('metadata', {})
            # Common keys for hidden size in different architectures
            hidden_size_keys = [
                'llama.embedding_length',  # Llama models
                'llama.rope.dimension_count',  # Sometimes used
                'hidden_size',  # Generic
                'n_embd',  # GPT-style
            ]
            for key in hidden_size_keys:
                if key in metadata:
                    return metadata[key]

            # Try to infer from tensor dimensions
            tensor_info = result.get('tensor_info', [])
            for tensor in tensor_info:
                dims = tensor.get('dimensions', [])
                name = ''  # We don't have name here, but could add if needed
                # Look for embedding or attention tensors
                if len(dims) >= 2 and 'embd' in name.lower():
                    return dims[-1]  # Last dimension is often hidden size

        return None

    @staticmethod
    def get_model_gqa_factor_from_gguf(model_path: str) -> Optional[float]:
        """
        Extract the GQA factor from a GGUF file.

        Args:
            model_path: Path to the GGUF model file

        Returns:
            GQA factor (1.0 for no GQA, <1 for GQA) or None if extraction fails
        """
        result = GGUFUtils.extract_gguf_parameters(model_path)
        if result:
            metadata = result.get('metadata', {})
            # Look for attention head info
            n_head_key = 'llama.attention.head_count'
            n_head_kv_key = 'llama.attention.head_count_kv'

            n_head = metadata.get(n_head_key)
            n_head_kv = metadata.get(n_head_kv_key)

            if n_head and n_head_kv:
                return n_head_kv / n_head

        # Default to 1.0 (no GQA) if not found
        return 1.0

    @staticmethod
    def extract_parameters_from_model_name(model_name: str) -> Optional[int]:
        """
        Extract the number of parameters from a model name string.

        Supports formats like:
        - "7B" -> 7_000_000_000
        - "0.5B" -> 500_000_000
        - "13B" -> 13_000_000_000
        - "1.5B" -> 1_500_000_000
        - "70B" -> 70_000_000_000

        Args:
            model_name: The model name string to parse

        Returns:
            Number of parameters as integer, or None if not found
        """
        import re

        # Pattern to match parameter counts like "0.5B", "7B", "13B", etc.
        # Matches optional decimal part followed by B
        pattern = r'(\d+(?:\.\d+)?)B'
        match = re.search(pattern, model_name, re.IGNORECASE)

        if match:
            param_str = match.group(1)
            try:
                # Convert to float first to handle decimals
                param_value = float(param_str)
                # Convert to billions
                parameters = int(param_value * 1_000_000_000)
                return parameters
            except ValueError:
                return None

        return None