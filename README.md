# Llama Smart Proxy

A smart proxy server for Llama models with agent-based request/response processing capabilities. This project provides a FastAPI-based proxy that can load and serve multiple Llama models concurrently, with extensible agent plugins for preprocessing requests and postprocessing responses.

## Features

- **Multi-Model Support**: Load and serve multiple Llama models simultaneously
- **Server Pool Management**: Efficient server pool with lazy loading and LRU eviction
- **Agent System**: Extensible plugin system for request/response processing
- **OpenAI-Compatible API**: Compatible with OpenAI Chat Completions API
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
  "server_pool": {
    "size": 2,
    "host": "localhost",
    "port_start": 8001
  },
  "models": {
    "llama-7b": {
      "repo": "TheBloke/Llama-2-7B-Chat-GGUF",
      "variant": "llama-2-7b-chat.Q4_K_M.gguf"
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

#### Chat Completions
```http
POST /chat/completions
```

Example request:
```json
{
  "model": "unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
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

## Configuration

### Server Pool
- `size`: Number of server instances to maintain
- `host`: Host for server instances
- `port_start`: Starting port number for servers

### Models
Models are specified with HuggingFace repository IDs and variant patterns:
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

```
llama-smart-proxy/
├── src/proxy/
│   ├── api.py              # FastAPI application and routes
│   ├── server_pool.py      # Server pool management
│   ├── agent_manager.py    # Agent plugin system
│   ├── model_resolver.py   # Model identifier resolution
│   ├── config.py           # Configuration loading
│   ├── types.py            # Pydantic models
│   └── common_imports.py   # Shared utilities
├── tests/                  # Unit tests
├── plugins/                # Agent plugins directory
├── config.json             # Configuration file
├── requirements.txt        # Python dependencies
├── main.py                 # Application entry point
└── main_sim.py             # E2E simulation script
```

## License

See LICENSE file for details.