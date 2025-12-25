import httpx
import logging

from fastapi import Request, Response
from src.frameworks_drivers.base_llm_service import BaseLLMService
from src.frameworks_drivers.model_resolver import ModelResolver
from src.frameworks_drivers.server_pool import ServerPool
from src.shared.protocols import LLMServiceProtocol

logger = logging.getLogger(__name__)


class LlamaCppLLMService(BaseLLMService, LLMServiceProtocol):
    def __init__(self, server_pool: ServerPool, model_resolver: ModelResolver, timeout: float = 30.0):
        super().__init__(timeout=timeout)
        self.server_pool = server_pool
        self.model_resolver = model_resolver

    async def generate_completion(self, request: dict) -> dict:
        model = request.get("model")
        if not model:
            raise ValueError("Model not specified in request")

        repo_id, _ = self.model_resolver.resolve(model)
        server = await self.server_pool.get_server_for_model(repo_id)
        if not server:
            raise RuntimeError(f"No available server for model {model}")

        url = f"http://127.0.0.1:{server.port}/v1/chat/completions"
        logger.info(f"Sending request to Llama.cpp: URL={url}, request={request}")
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=request, timeout=self.timeout)
            logger.info(f"Llama.cpp response status: {response.status_code}, text: {response.text[:200]}")
            response.raise_for_status()
            return response.json()

    async def forward_request(self, path: str, request: Request) -> Response:
        # Parse the request body to get the model
        request_data = await request.json()
        model = request_data.get("model")
        if not model:
            raise ValueError("Model not specified in forwarding request")

        repo_id, _ = self.model_resolver.resolve(model)
        server = await self.server_pool.get_server_for_model(repo_id)
        if not server:
            raise RuntimeError(f"No available server for model {model}")

        target_url = f"http://127.0.0.1:{server.port}/{path}"
        return await self._forward_request(target_url, request)
