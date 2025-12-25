# Data Model: Clean Architecture Conversion and Ollama Preparation

**Date**: 2025-12-25

## Entities

### Model
Represents an LLM model configuration.

**Fields**:
- `id`: str - Unique identifier (e.g., "llama-7b")
- `repo`: str - HuggingFace repository (e.g., "TheBloke/Llama-2-7B-Chat-GGUF")
- `variant`: str - Model variant/filename (e.g., "llama-2-7b-chat.Q4_K_M.gguf")
- `backend`: str - Backend type ("llama.cpp" or "ollama")

**Validation Rules**:
- `id` must be non-empty string
- `repo` must be valid HuggingFace format
- `backend` must be one of allowed values

**Relationships**:
- Used by requests

### Server
Represents a backend server instance.

**Fields**:
- `id`: str - Unique identifier
- `host`: str - Host address
- `port`: int - Port number
- `model_id`: str - Associated model ID
- `status`: str - Current status ("running", "stopped", "error")
- `process`: Optional[Process] - Server process handle

**Validation Rules**:
- `port` must be valid port number (1-65535)
- `status` must be one of allowed values

**Relationships**:
- Belongs to Model (many-to-one)

**State Transitions**:
- stopped → running (start)
- running → stopped (stop)
- running → error (failure)

### Agent
Represents a processing plugin.

**Fields**:
- `name`: str - Plugin name
- `enabled`: bool - Whether active
- `hooks`: List[str] - Available hooks ("request", "response")

**Validation Rules**:
- `name` must be non-empty
- `hooks` must contain valid hook names

**Relationships**:
- Processes messages

### Message
Represents a chat message.

**Fields**:
- `role`: str - Message role ("user", "assistant", "system")
- `content`: str - Message content

**Validation Rules**:
- `role` must be one of allowed values
- `content` must be non-empty string

**Relationships**:
- Part of requests

## Relationships Diagram

```
Model 1..* ---- Server
Model 1 ---- * Message (via requests)
Agent * ---- * Message (processes)
```</content>
