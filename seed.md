# Llama Smart Proxy

**Technical Specification**

---

## 1. Introduction

### 1.1 Project Overview

Llama Smart Proxy is a lightweight, extensible proxy server for managing multiple `llama.cpp` instances through a fixed-size server pool.
It exposes **llama.cpp-compatible OpenAI-style APIs** and extends functionality via a **dynamic agent (plugin) architecture** without modifying the underlying `llama.cpp` implementation.

The proxy supports:

* Dynamic **model loading based on client requests**
* Lazy initialization and reuse of llama.cpp servers
* Modular **agent plugins** for request and response processing
* **Prompt-driven agent activation** using slash commands
* Simple configuration and deployment

---

### 1.2 Objectives

* Manage a pool of llama.cpp servers with minimal operational complexity
* Dynamically load models specified by clients at request time
* Enable request/response customization via modular agents
* Allow per-request agent activation and ordering
* Maintain compatibility with llama.cpp/OpenAI-style APIs

---

### 1.3 Core Concepts

| Concept                     | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| Server Pool                 | Fixed-capacity pool of llama.cpp servers, lazily initialized |
| Model-Driven Initialization | Server model is determined by the client `model` field       |
| Agents                      | Plugins that modify requests and/or responses                |
| Agent Chain                 | Ordered sequence of agents activated per request             |
| Slash Commands              | Prompt prefixes that activate agents dynamically             |
| Transparency                | Core proxy preserves message format and structure; agents may modify content |

---

## 2. Architecture

### 2.1 High-Level Design

The proxy consists of a **core routing layer** and **extensible processing layers**.

**Responsibilities**:

* Server lifecycle management
* Model resolution and loading
* Request routing
* Agent discovery and execution

---

### 2.2 Data Flow (Authoritative)

1. Client sends an OpenAI-style request to the proxy
2. Proxy extracts the `model` field from the request body
3. Proxy resolves the model identifier into a concrete load configuration
4. Proxy selects a server from the pool:

   * Reuses an already-initialized server if model matches
   * Otherwise initializes a server with the requested model
5. Proxy parses the prompt for slash commands (if applicable)
6. Proxy builds the agent execution chain
7. Request is processed through request-phase agents
8. Request is forwarded to the selected llama.cpp server
9. Response is processed through response-phase agents
10. Final response is returned to the client

---

### 2.3 Components

* **Server Pool Manager**

  * Maintains a fixed-capacity pool
  * Performs lazy initialization and health checks

* **Plugin Manager**

  * Discovers agents from plugin directories
  * Loads agent configurations

* **Agent Chain Executor**

  * Applies agents in deterministic order
  * Executes both request and response hooks

* **API Layer**

  * Exposes llama.cpp-compatible endpoints (e.g. `/chat/completions`)

---

### 2.4 Technology Stack

* Language: Python 3.12+
* Framework: FastAPI
* LLM Backend: `llama_cpp` Python library
* Configuration: JSON

---

## 3. Server Pool Management

### 3.1 Pool Characteristics

* Pool size is **fixed** and defined in configuration
* Servers are **lazily initialized**
* Each server holds **one model at a time**
* Servers are reused only if the loaded model matches the request

---

### 3.2 Server Selection Rules

1. Prefer a healthy server with the requested model loaded
2. Otherwise select an uninitialized server
3. If pool is full and no compatible server exists:

   * Behavior is implementation-defined (error or eviction)

---

### 3.3 Health Monitoring

* Basic health status per server
* `/health` endpoint reports:

  * Pool capacity
  * Server health
  * Loaded model identifiers

---

## 4. Model Loading From Client Requests

### 4.1 Client-Specified Model Field

Clients must specify the model using the `model` field:

```json
{
  "model": "unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
  "messages": [...]
}
```

---

### 4.2 Supported Model Identifier Formats

1. **Hugging Face repo + variant (recommended)**

   ```
   repo_id:variant
   ```

   Example:

   ```
   unsloth/Qwen3-0.6B-GGUF:Q4_K_M
   ```

2. **Hugging Face repo only**

   ```
   repo_id
   ```

   A default GGUF variant is applied.

---

### 4.3 Model Resolution Rules

* If `:` is present:

  * Left side → `repo_id`
  * Right side → `variant`
* Variant is converted into a filename or pattern (e.g. `*Q4_K_M.gguf`)
* Resolution is performed **before agent execution**

---

### 4.4 Server Initialization

Servers are initialized using `llama-cpp-python` semantics:

* Model is loaded using `from_pretrained`
* Configuration is derived solely from the request
* Initialization occurs once per server/model pairing

---

### 4.5 Error Conditions

| Condition             | Behavior                                |
| --------------------- | --------------------------------------- |
| Missing `model`       | Client error (400)                      |
| Invalid identifier    | Model load error                        |
| Download/load failure | Server marked unhealthy; error returned |

---

## 5. Plugin and Agent Architecture

### 5.1 Plugin Discovery

Agents are discovered at startup by scanning `plugins_dir`.

**Directory structure**:

```
plugins/
├── agent_name/
│   ├── agent.py
│   └── config.json
```

* Directory name defines agent alias (e.g. `/rag`)
* Agents are loaded once at startup

---

### 5.2 Agent Configuration

Example `config.json`:

```json
{
  "description": "RAG agent for context augmentation",
  "enabled": true,
  "end_point": "/chat/completions"
}
```

| Field     | Meaning                        |
| --------- | ------------------------------ |
| enabled   | Enables/disables agent         |
| end_point | Endpoint this agent applies to |

---

### 5.3 Agent Lifecycle

Each agent may implement:

* Request hook
* Response hook

Agents are executed sequentially and deterministically.

---

## 6. Agent Chain Execution

### 6.1 Slash Command Activation

Agents are activated by prefixing the prompt:

```
/rag /parallel Hello world
```

Rules:

* Slash commands are parsed **left to right**
* Order defines execution order
* Unknown commands are ignored
* Slash commands apply only to prompt-based endpoints

---

### 6.2 Execution Phases

1. **Request Phase**

   * Agents modify the outgoing request

2. **Response Phase**

   * Same agents run again in the same order
   * Each agent receives the previous agent’s output

---

### 6.3 Independence From Model Loading

* Model selection and loading occur **before** agent execution
* Agents do not influence server or model selection

---

## 7. API Behavior

### 7.1 Supported Endpoints

* `/chat/completions`
* Other llama.cpp-compatible endpoints

---

### 7.2 Non-Prompt Endpoints

* Slash commands are ignored
* No dynamic agent activation occurs

---

## 8. Configuration

### 8.1 Main Configuration File

```json
{
  "plugins_dir": "./plugins",
  "server_pool": {
    "num_servers": 2
  }
}
```

---

## 9. Error Handling

* Agent exceptions are logged
* Server failures are isolated
* Basic fallback error responses are returned
* No proxy-wide crash on individual failures

---

## 10. Deployment

* Direct Python execution
* No external orchestration required
* Model downloads handled by llama.cpp backend

---

## 11. Testing and Observability

* Integration testing for:

  * Server pool reuse
  * Model loading
  * Agent ordering
* Logging includes:

  * Model resolution
  * Server selection
  * Agent execution order
* `/health` endpoint for operational insight

---

## 12. Guarantees

* Deterministic agent execution order
* One model per server instance
* Client-controlled model selection
* No modification of llama.cpp internals
* Extensibility via isolated plugins
* Proxy transparency: Core proxy does not alter input/output message formats or structures

