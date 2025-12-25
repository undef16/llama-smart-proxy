# Quick Start: Clean Architecture Conversion and Ollama Preparation

**Date**: 2025-12-25

## Overview

This feature converts the Llama Smart Proxy to Clean Architecture and prepares for Ollama backend integration.

## Architecture Overview

The codebase is organized in Clean Architecture layers:

- **Entities**: Core business objects (Model, Server, Agent, Message)
- **Use Cases**: Application business logic (ProcessChatCompletion, GetHealth)
- **Interface Adapters**: Controllers and external interfaces
- **Frameworks & Drivers**: External frameworks and implementations

## Key Changes

### 1. Code Organization
- Existing code in `src/proxy/` will be refactored into layers
- New packages: `entities/`, `use_cases/`, `interface_adapters/`, `frameworks_drivers/`

### 2. Dependency Injection
- Manual dependency injection for testability
- Protocols/interfaces for dependency inversion

### 3. Backend Abstraction
- LLM backends abstracted via `LLMService` interface
- Support for both llama.cpp and Ollama implementations

## Migration Steps

1. **Install dependencies** (if any new ones)
2. **Run existing tests** to ensure no regressions
3. **Update configuration** for backend selection
4. **Test with both backends**

## Configuration

Add backend selection to `config.json`:

```json
{
  "backend": "ollama",  // or "llama.cpp"
  "ollama": {
    "host": "http://localhost:11434",
    "models": ["llama2", "codellama"]
  }
}
```

## Testing

- Run `python -m pytest tests/` for unit tests
- Run `python main_sim.py` for E2E simulation
- Verify agent plugins still work
- Test both llama.cpp and Ollama backends

## API Compatibility

The API adapts to the selected backend:

- **llama.cpp backend**: OpenAI-compatible API
- **Ollama backend**: Ollama-compatible API

Example with llama.cpp:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Example with Ollama:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Health Check

```bash
curl http://localhost:8000/health
```

Returns server pool status for monitoring.</content>
