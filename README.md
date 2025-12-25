# Llama Smart Proxy

A smart proxy server for Llama models with agent-based request/response processing capabilities, built using Clean Architecture principles. This project provides a FastAPI-based proxy that can load and serve multiple Llama models concurrently, with extensible agent plugins for preprocessing requests and postprocessing responses. It supports multiple LLM backends including llama.cpp and Ollama.

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

## Installation

1. Clone the repository:
```bash
git clone https://github.com/undef16/llama-smart-proxy.git
cd llama-smart-proxy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the proxy by editing `config.json`:
```json
{
  "backend": "ollama",  // or "llama.cpp"
  "server_pool": {
    "size": 2,
    "host": "localhost",
    "port_start": 8001
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
│   ├── entities/           # Core business objects (Model, Server, Agent, Message)
│   ├── use_cases/          # Application business logic (ProcessChatCompletion, GetHealth)
│   ├── interface_adapters/ # Controllers and external interfaces (API, ChatController)
│   └── frameworks_drivers/ # External frameworks (LLM services, Model repository)
├── tests/                  # Unit tests
├── plugins/                # Agent plugins directory
├── config.json             # Configuration file
├── requirements.txt        # Python dependencies
├── main.py                 # Application entry point
└── main_sim.py             # E2E simulation script
```

## License

See LICENSE file for details.