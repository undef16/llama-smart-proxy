import asyncio
import httpx
import logging
import time

from fastapi import Request, Response
from src.frameworks_drivers.base_llm_service import BaseLLMService
from src.shared.protocols import LLMServiceProtocol

logger = logging.getLogger(__name__)


class OllamaLLMService(BaseLLMService, LLMServiceProtocol):
    def __init__(self, host: str = "localhost", port: int = 11434, timeout: float = 30.0, max_retries: int = 3):
        super().__init__(timeout=timeout)
        self.host = host
        self.port = port
        self.max_retries = max_retries

    async def generate_completion(self, request: dict) -> dict:
        url = f"http://{self.host}:{self.port}/v1/chat/completions"
        logger.info(f"Sending request to Ollama: URL={url}, request={request}")
        start_time = time.time()
        timeout = httpx.Timeout(read=self.timeout, connect=10.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(url, json=request)
                elapsed = time.time() - start_time
                logger.info(f"Ollama response status: {response.status_code}, elapsed: {elapsed:.2f}s, text: {response.text[:200]}")
                response.raise_for_status()
                return response.json()
            except httpx.ReadTimeout:
                elapsed = time.time() - start_time
                logger.error(f"Read timeout occurred for request to {url} after {elapsed:.2f}s")
                raise Exception("Request timed out while reading response from Ollama")
            except httpx.TimeoutException as e:
                elapsed = time.time() - start_time
                logger.error(f"Timeout occurred after {elapsed:.2f}s: {e}")
                raise Exception(f"Request to Ollama timed out: {e}")
            except httpx.HTTPStatusError as e:
                elapsed = time.time() - start_time
                logger.error(f"HTTP error after {elapsed:.2f}s: {e}")
                raise
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Unexpected error after {elapsed:.2f}s: {e}")
                raise

    async def forward_request(self, path: str, request: Request) -> Response:
        target_url = f"http://{self.host}:{self.port}/{path}"
        return await self._forward_request(target_url, request)
