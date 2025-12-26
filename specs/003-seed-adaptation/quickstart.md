# Quickstart: GPU Integration for Llama-Smart-Proxy

## Overview
This guide provides instructions for setting up and using the GPU integration features in Llama-Smart-Proxy.

## Prerequisites
1. NVIDIA GPU with CUDA support
2. NVIDIA drivers installed
3. CUDA toolkit installed
4. Python 3.8 or higher
5. pynvml library (pip install pynvml)

## Installation

### 1. Install pynvml
```bash
pip install pynvml
```

### 2. Verify GPU Setup
```bash
nvidia-smi
```

## Configuration

### 1. Update Configuration File
Add GPU-related settings to your `config.json`:

```json
{
  "backend": "llama.cpp",
  "server_pool": {
    "size": 2,
    "host": "localhost",
    "port_start": 11601,
    "gpu_layers": 99,
    "request_timeout": 600,
    "health_timeout": 5.0
  },
  "gpu": {
    "enabled": true,
    "enable_gpu_monitoring": true,
    "allocation_strategy": "single-gpu-preferred",
    "gpu_allocation_strategy": "single-gpu-preferred",
    "monitoring_interval": 5.0,
    "cpu_fallback": true
  },
  "ollama": {
    "host": "localhost",
    "port": 11434,
    "timeout": 300.0
  },
  "server": {
    "host": "0.0.0",
    "port": 11555
  },
  "agents": []
}
```

### 2. New Configuration Options
- `gpu.enabled`: Enables GPU monitoring and allocation features
- `gpu.enable_gpu_monitoring`: Enables GPU monitoring functionality
- `gpu.allocation_strategy`: Sets the GPU allocation strategy ("single-gpu-preferred" or "distribute")
- `gpu.gpu_allocation_strategy`: Alternative GPU allocation strategy setting
- `gpu.monitoring_interval`: GPU monitoring interval in seconds
- `gpu.cpu_fallback`: Enable CPU fallback when GPU is unavailable

## Usage

### 1. Start the Proxy with GPU Support
```bash
python main.py
```

### 2. Send Requests
The proxy will automatically select appropriate GPU resources when loading models based on available GPU memory and utilization.

### 3. Check GPU Status
Query the health endpoint to see GPU status information:
```bash
curl http://localhost:1555/health
```

## API Changes

### Enhanced Health Endpoint Response
The health endpoint now includes GPU information:

```json
{
  "servers": [
    {
      "id": "server_0",
      "host": "localhost",
      "port": 11601,
      "model_id": "microsoft/Phi-3-mini-4k-instruct",
      "status": "running",
      "process": 12345,
      "gpu_assignment": {
        "gpu_ids": [0],
        "estimated_vram_required": 2.1,
        "actual_vram_used": 2.3
      }
    }
  ],
  "gpu_pool_status": {
    "total_gpus": 2,
    "available_gpus": 1,
    "total_vram": 24.0,
    "available_vram": 21.7,
    "gpu_list": [
      {
        "id": 0,
        "name": "GeForce RTX 4080",
        "total_memory": 16.0,
        "free_memory": 13.7,
        "used_memory": 2.3,
        "utilization": 15.2,
        "temperature": 42.0,
        "power_usage": 85.5,
        "assigned_models": ["microsoft/Phi-3-mini-4k-instruct"],
        "compute_capability": "8.9"
      },
      {
        "id": 1,
        "name": "GeForce RTX 3080",
        "total_memory": 8.0,
        "free_memory": 8.0,
        "used_memory": 0.0,
        "utilization": 0.0,
        "temperature": 35.0,
        "power_usage": 35.0,
        "assigned_models": [],
        "compute_capability": "8.6"
      }
    ],
    "allocation_strategy": "single-gpu-preferred"
  }
}
```

## Troubleshooting

### GPU Not Detected
- Verify NVIDIA drivers are installed: `nvidia-smi`
- Check pynvml installation: `python -c "import pynvml; print('pynvml available')"`
- Ensure CUDA is properly installed

### VRAM Estimation Issues
- The system uses conservative estimates based on model parameters and quantization
- If experiencing OOM errors, consider reducing the number of GPU layers or using a smaller model

### CPU-Only Fallback
- The system will automatically fall back to CPU-only operation when no GPUs are available
- Check logs for warnings about GPU unavailability