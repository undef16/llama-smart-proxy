from src.entities.model import Model
from src.entities.server import Server
from src.shared.protocols import ModelRepositoryProtocol as ModelRepositoryProtocol, ServerDTO
from src.frameworks_drivers.model_resolver import ModelResolver
from src.utils.gguf_utils import GGUFUtils



class ModelRepository(ModelRepositoryProtocol):
    def __init__(self):
        self.models: dict[str, Model] = {}
        self.servers: list[Server] = []

    def get_model(self, model_id: str) -> Model:

        if model_id in self.models:
            return self.models[model_id]

        # Determine if this is a local GGUF file or HuggingFace repo
        if model_id.endswith('.gguf'):
            # Local GGUF file
            repo = "local"
            variant = model_id[:-5]  # Remove .gguf extension
        else:
            # HuggingFace repository format
            try:
                repo_id, filename_pattern = ModelResolver.resolve(model_id)
                repo = repo_id
                # Extract variant from filename_pattern
                if filename_pattern.startswith('*') and filename_pattern.endswith('.gguf'):
                    variant = filename_pattern[1:-5]  # Remove * and .gguf
                else:
                    variant = filename_pattern
            except ValueError:
                # Fallback: treat as repo with default variant
                repo = model_id
                variant = "Q4_K_M"

        # Validate repo format
        import re
        if not (re.match(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$", repo) or repo == "local"):
            raise KeyError(f"Model {model_id} not found in repository")

        # Assume backend is llama.cpp for now, as this is the current backend
        backend = "llama.cpp"

        # Try to extract parameter count from model name
        parameters = GGUFUtils.extract_parameters_from_model_name(model_id)

        model = Model(
            id=model_id,
            repo=repo,
            variant=variant,
            backend=backend,
            parameters=parameters
        )
        self.models[model_id] = model
        return model

    def get_all_models(self) -> list[Model]:
        return list(self.models.values())

    def get_servers_for_model(self, model_id: str) -> list[ServerDTO]:
        return [
            {
                "id": server.id,
                "host": server.host,
                "port": server.port,
                "model_id": server.model_id,
                "status": server.status,
                "process": server.process,
                "gpu_assignment": server.gpu_assignment.model_dump() if server.gpu_assignment else None
            }
            for server in self.servers if server.model_id == model_id
        ]
