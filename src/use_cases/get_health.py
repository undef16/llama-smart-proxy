from typing import Any

from src.shared.protocols import ModelRepositoryProtocol


class GetHealth:
    def __init__(self, model_repository: ModelRepositoryProtocol):
        self.model_repository = model_repository

    def execute(self) -> dict[str, Any]:
        models = self.model_repository.get_all_models()
        servers = []
        for model in models:
            model_servers = self.model_repository.get_servers_for_model(model.id)
            servers.extend(model_servers)
        return {"servers": [server.model_dump() for server in servers]}
