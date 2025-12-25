import asyncio

import httpx
from fastapi import FastAPI, Request, Response

from src.interface_adapters.chat_controller import ChatController
from src.interface_adapters.health_controller import HealthController


class API:
    def __init__(self, chat_controller: ChatController, health_controller: HealthController):
        self.chat_controller = chat_controller
        self.health_controller = health_controller
        self.app = FastAPI(title="Llama Smart Proxy", version="0.1.0")
        self._register_routes()

    def _register_routes(self):
        self.app.post("/v1/chat/completions")(self.chat_completions)
        self.app.post("/v1/completions")(self.chat_completions)
        self.app.post("/chat/completions")(self.chat_completions)
        self.app.get("/health")(self.health)
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])(
            self._forward_request,
        )

    async def chat_completions(self, request: dict) -> dict:
        return await self.chat_controller.chat_completions(request)

    def health(self):
        return self.health_controller.health()

    async def _forward_request(self, path: str, request: Request) -> Response:
        # Get the LLM service from the chat controller's use case
        llm_service = self.chat_controller.process_chat_completion_use_case.llm_service

        # Delegate forwarding to the service
        return await llm_service.forward_request(path, request)
