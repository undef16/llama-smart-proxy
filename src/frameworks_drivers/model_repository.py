
from src.entities.model import Model
from src.entities.server import Server
from src.shared.protocols import ModelRepositoryProtocol as ModelRepositoryProtocol


class ModelRepository(ModelRepositoryProtocol):
    def __init__(self):
        self.models: dict[str, Model] = {}
        self.servers: list[Server] = []

    def get_model(self, model_id: str) -> Model:
        return self.models[model_id]

    def get_all_models(self) -> list[Model]:
        return list(self.models.values())

    def get_servers_for_model(self, model_id: str) -> list[Server]:
        return [server for server in self.servers if server.model_id == model_id]
