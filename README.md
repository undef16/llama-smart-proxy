# Llama Smart Proxy

A smart proxy server for Llama models with agent-based request/response processing capabilities, built using Clean Architecture principles. This project provides a FastAPI-based proxy that can load and serve multiple Llama models concurrently, with extensible agent plugins for preprocessing requests and postprocessing responses. It supports multiple LLM backends including llama.cpp and Ollama, with intelligent GPU resource management and allocation.

## Features

- **Clean Architecture**: Organized in layers (Entities, Use Cases, Interface Adapters, Frameworks & Drivers) for maintainability and testability
- **Multi-Backend Support**: Pluggable LLM backends (llama.cpp and Ollama) for flexible model management
- **Multi-Model Support**: Load and serve multiple Llama models simultaneously
- **Server Pool Management**: Efficient server pool with lazy loading and LRU eviction
- **Agent System**: Extensible plugin system for request/response processing
- **OpenAI-Compatible API**: Compatible with OpenAI Chat Completions API (llama.cpp backend)
- **Ollama-Compatible API**: Native Ollama API support for streamlined model operations
- **Async Processing**: Built with asyncio for high performance
- **Health Monitoring**: Built-in health checks and server status monitoring
- **GPU Resource Management**: Intelligent GPU detection, monitoring, and allocation with automatic CPU fallback
- **VRAM Estimation**: Automatic VRAM requirement calculation for GGUF models
- **Multi-GPU Support**: Distribute large models across multiple GPUs or prefer single-GPU allocation

## Installation

### Prerequisites

- Python 3.12+
- For GPU support: NVIDIA GPU with CUDA compute capability 6.1+, CUDA drivers, and pynvml library
- For CPU-only operation: No additional requirements

### Setup

1. Clone the repository:
```bash
git clone https://github.com/undef16/llama-smart-proxy.git
cd llama-smart-proxy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For GPU support, ensure NVIDIA drivers are installed and CUDA is available:
```bash
nvidia-smi  # Verify GPU detection
```

3. Configure the proxy by editing `config.json`:
```json
{
  "backend": "ollama",  // or "llama.cpp"
  "server_pool": {
    "size": 2,
    "host": "localhost",
    "port_start": 8001,
    "gpu_layers": 99
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
    "host": "http://localhost:11434",
    "models": ["llama2", "codellama"]
  },
  "llama_cpp": {
    "models": {
      "llama-7b": {
        "repo": "TheBloke/Llama-2-7B-Chat-GGUF",
        "variant": "llama-2-7b-chat.Q4_K_M.gguf"
      }
    }
  },
  "agents": ["rag", "parallel"]
}
```

## Usage

### Starting the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`.

### API Endpoints

#### Chat Completions (OpenAI-compatible, llama.cpp backend)
```http
POST /v1/chat/completions
```

Example request:
```json
{
  "model": "llama2",
  "messages": [
    {
      "role": "user",
      "content": "Tell me a joke about programming."
    }
  ],
  "temperature": 0.7,
  "max_tokens": 100
}
```

#### Chat Completions (Ollama-compatible, Ollama backend)
```http
POST /api/chat
```

Example request:
```json
{
  "model": "llama2",
  "messages": [
    {
      "role": "user",
      "content": "Tell me a joke about programming."
    }
  ],
  "stream": false
}
```

#### Health Check
```http
GET /health
```

Returns server pool status and health information.

### Agent System

The proxy supports agent plugins that can preprocess requests and postprocess responses. Agents are loaded from the `plugins/` directory.

To use an agent in a request, include slash commands in the user message:
```
User: /rag /parallel Tell me about machine learning
```

Available agents are configured in `config.json`.

## Migration from Previous Version

The codebase has been refactored to follow Clean Architecture principles and now supports multiple LLM backends. To migrate from the previous version:

1. **Update Configuration**: Add `backend` selection and reorganize model configurations under `ollama` or `llama_cpp` sections as shown above.

2. **API Endpoint Changes**:
   - For llama.cpp backend: Use `/v1/chat/completions` (OpenAI-compatible)
   - For Ollama backend: Use `/api/chat` (Ollama-compatible)

3. **Code Structure**: The internal code has been reorganized into Clean Architecture layers. If you have custom plugins or extensions, ensure they interface with the new use case and entity layers.

4. **Dependencies**: No new dependencies required for the migration, but ensure Ollama is installed and running if switching backends.

5. **Testing**: Run existing tests and E2E simulation to verify functionality.

## Configuration

### Backend Selection
- `backend`: Choose the LLM backend ("llama.cpp" or "ollama")

### Server Pool (llama.cpp backend only)
- `size`: Number of server instances to maintain
- `host`: Host for server instances
- `port_start`: Starting port number for servers
- `gpu_layers`: Number of layers to offload to GPU (0 = CPU only)

### GPU Configuration
- `enabled`: Enable GPU monitoring and allocation
- `enable_gpu_monitoring`: Enable GPU monitoring functionality
- `allocation_strategy`: GPU allocation strategy ("single-gpu-preferred" or "distribute")
- `gpu_allocation_strategy`: Alternative GPU allocation strategy setting
- `monitoring_interval`: GPU monitoring interval in seconds
- `cpu_fallback`: Enable CPU fallback when GPU is unavailable

### Ollama Configuration
- `host`: Ollama server URL (default: "http://localhost:11434")
- `models`: List of available Ollama models

### Llama.cpp Configuration
- `models`: Dictionary of model configurations with HuggingFace repository IDs and variant patterns:
```json
{
  "model-name": {
    "repo": "organization/model-repo",
    "variant": "model-variant.gguf"
  }
}
```

### Agents
List of enabled agent plugin names:
```json
["agent1", "agent2"]
```

## Deployment

### GPU Environment Setup

For production deployment with GPU support:

1. **Install NVIDIA Drivers and CUDA**:
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install nvidia-driver-XXX cuda-toolkit-XX-X

   # Verify installation
   nvidia-smi
   nvcc --version
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pynvml
   ```

3. **Configure GPU Settings**:
   ```json
   {
     "gpu": {
       "enabled": true,
       "enable_gpu_monitoring": true,
       "allocation_strategy": "single-gpu-preferred",
       "monitoring_interval": 5.0,
       "cpu_fallback": true
     }
   }
   ```

4. **System Requirements**:
   - NVIDIA GPU with CUDA compute capability 6.1+
   - Minimum 8GB VRAM for small models
   - 16GB+ VRAM recommended for larger models
   - Sufficient system RAM (2x model size recommended)

### Docker Deployment

```dockerfile
FROM nvidia/cuda:12.2-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip install -r requirements.txt && pip install pynvml

# Copy application
COPY . /app
WORKDIR /app

# Run with GPU support
CMD ["python", "main.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-smart-proxy
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: proxy
        image: your-registry/llama-smart-proxy:latest
        resources:
          limits:
            nvidia.com/gpu: 1  # Request GPU resources
        env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: "0"
```

## GPU Troubleshooting

### Common Issues

#### GPU Not Detected
**Symptoms**: Health endpoint shows no GPUs, models run on CPU only
**Solutions**:
1. Verify NVIDIA drivers are installed: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Ensure pynvml is installed: `pip install pynvml`
4. Check GPU monitoring is enabled in config: `"enable_gpu_monitoring": true`

#### Insufficient GPU Memory
**Symptoms**: Model loading fails with GPU allocation errors
**Solutions**:
1. Check available VRAM: `nvidia-smi`
2. Reduce `gpu_layers` in server_pool config
3. Use smaller model variants (Q4_K_M instead of Q8_0)
4. Enable multi-GPU distribution: `"allocation_strategy": "distribute"`

#### GPU Monitoring Disabled
**Symptoms**: GPU status not shown in health endpoint
**Solutions**:
1. Set `"enable_gpu_monitoring": true` in config
2. Check pynvml availability
3. Verify GPU hardware is accessible

#### High GPU Memory Usage
**Symptoms**: GPU memory not freed after model unloading
**Solutions**:
1. Ensure proper server shutdown
2. Check for GPU context cleanup in logs
3. Restart server pool if memory leaks persist

### Performance Tuning

- **Single GPU Preferred**: Use for models that fit in one GPU's memory
- **Multi-GPU Distribution**: Use for large models requiring multiple GPUs
- **CPU Fallback**: Automatically enabled when GPUs unavailable
- **Monitoring Interval**: Adjust `monitoring_interval` based on needs (lower = more frequent updates)

## Development

### Running Tests
```bash
python -m pytest tests/
```

### E2E Simulation
```bash
python main_sim.py
```

This script starts the proxy and performs an end-to-end test with a real chat completion request.

## Project Structure

The codebase follows Clean Architecture principles, organized into layers with dependencies pointing inward:

```
llama-smart-proxy/
├── src/
│   ├── entities/           # Core business objects (Model, Server, Agent, Message, GPU, GPU Assignment, GPU Pool Status, Performance Monitor)
│   ├── use_cases/          # Application business logic (ProcessChatCompletion, GetHealth)
│   ├── interface_adapters/ # Controllers and external interfaces (API, ChatController, Health Controller)
│   ├── frameworks_drivers/ # External frameworks (LLM services, Model repository, GPU monitoring, allocation, and management)
│   └── utils/              # Utility functions (VRAM estimation, GGUF utilities)
├── specs/                  # Feature specifications and documentation
├── tests/                  # Unit tests
├── plugins/                # Agent plugins directory
├── config.json             # Configuration file
├── requirements.txt        # Python dependencies
├── main.py                 # Application entry point
└── main_sim.py             # E2E simulation script
```

## License

See LICENSE file for details.