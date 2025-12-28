import asyncio

import httpx
from fastapi import Depends, FastAPI, Request, Response

from src.interface_adapters.chat_controller import ChatController
from src.interface_adapters.health_controller import HealthController


class API:
    def __init__(self, config, server_pool, model_resolver, agent_manager, model_repository, gpu_monitor):
        self.config = config
        self.server_pool = server_pool
        self.model_resolver = model_resolver
        self.agent_manager = agent_manager
        self.model_repository = model_repository
        self.gpu_monitor = gpu_monitor
        self.app = FastAPI(title="Llama Smart Proxy", version="0.1.0")

        # Dependency functions for per-request instances
        self.get_chat_controller = lambda: self._create_chat_controller()
        self.get_health_controller = lambda: self._create_health_controller()
        self.get_llm_service = lambda: self._create_llm_service()

        self._register_routes()

    def _register_routes(self):
        # Register routes with dependencies
        async def chat_completions_handler(request: dict, controller=Depends(self.get_chat_controller)) -> dict:
            return await controller.chat_completions(request)

        def health_handler(controller=Depends(self.get_health_controller)):
            return controller.health()

        async def forward_request_handler(path: str, request: Request, llm_service=Depends(self.get_llm_service)) -> Response:
            return await llm_service.forward_request(path, request)

        self.app.post("/v1/chat/completions")(chat_completions_handler)
        self.app.post("/v1/completions")(chat_completions_handler)
        self.app.post("/chat/completions")(chat_completions_handler)
        self.app.get("/health")(health_handler)
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])(
            forward_request_handler,
        )

    def _create_chat_controller(self):
        from src.frameworks_drivers.llm_service_factory import LLMServiceFactory
        from src.use_cases.process_chat_completion import ProcessChatCompletion

        llm_service_factory = LLMServiceFactory(self.config.model_dump(), self.server_pool, self.model_resolver)
        llm_service = llm_service_factory.create_service()
        process_chat_completion = ProcessChatCompletion(llm_service, self.agent_manager)
        return ChatController(process_chat_completion)

    def _create_health_controller(self):
        from src.use_cases.get_health import GetHealth

        get_health = GetHealth(self.model_repository, self.gpu_monitor, self.config)
        return HealthController(get_health)

    def _create_llm_service(self):
        from src.frameworks_drivers.llm_service_factory import LLMServiceFactory

        llm_service_factory = LLMServiceFactory(self.config.model_dump(), self.server_pool, self.model_resolver)
        return llm_service_factory.create_service()

