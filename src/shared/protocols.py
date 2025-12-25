from typing import Protocol, Optional, Any, TypedDict


class ModelDTO(TypedDict):
    id: str
    repo: str
    variant: Optional[str]
    backend: str


class ServerDTO(TypedDict):
    id: str
    host: str
    port: int
    model_id: str
    status: str
    process: Optional[int]


class ModelRepositoryProtocol(Protocol):
    def get_model(self, model_id: str) -> ModelDTO: ...

    def get_all_models(self) -> list[ModelDTO]: ...

    def get_servers_for_model(self, model_id: str) -> list[ServerDTO]: ...


class LLMServiceProtocol(Protocol):
    async def generate_completion(self, request: dict) -> dict: ...

    async def forward_request(self, path: str, request) -> Any: ...


class BackendCheckerProtocol(Protocol):
    def check_availability(self, model: Optional[str] = None) -> bool: ...

    def get_forwarding_endpoints(self) -> list[dict]: ...

    def get_simulation_model(self) -> str: ...


class AgentManagerProtocol(Protocol):
    def parse_slash_commands(self, content: str) -> list[str]: ...

    def build_agent_chain(self, agent_names: list[str]) -> list: ...

    def execute_request_hooks(self, request: dict, agent_chain: list) -> dict: ...

    def execute_response_hooks(self, response: dict, agent_chain: list) -> dict: ...
