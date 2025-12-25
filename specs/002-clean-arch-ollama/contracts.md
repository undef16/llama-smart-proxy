# API Contracts: Clean Architecture Conversion and Ollama Preparation

**Date**: 2025-12-25

## API Endpoints

### POST /v1/chat/completions
Generate chat completions using LLM models.

**Request Body**:
```json
{
  "model": "string",
  "messages": [
    {
      "role": "user|assistant|system",
      "content": "string"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 100
}
```

**Response**:
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "response text"
      }
    }
  ]
}
```

### GET /health
Check server pool health status.

**Response**:
```json
{
  "servers": [
    {
      "model": "string",
      "status": "running|stopped|error",
      "port": 8001
    }
  ]
}
```

## Interface Contracts

### Model Repository Interface
```python
class ModelRepository(Protocol):
    def get_model(self, model_id: str) -> Model:
        ...
    
    def save_model(self, model: Model) -> None:
        ...
```

### LLM Service Interface
```python
class LLMService(Protocol):
    async def generate_completion(self, request: dict) -> dict:
        ...
```

### Agent Manager Interface
```python
class AgentManagerInterface(Protocol):
    def parse_slash_commands(self, content: str) -> List[str]:
        ...
    
    def execute_request_hooks(self, request: dict, agent_chain: List) -> dict:
        ...
    
    def execute_response_hooks(self, response: dict, agent_chain: List) -> dict:
        ...
```</content>
