import httpx
import logging

from fastapi import Request, Response

logger = logging.getLogger(__name__)


class BaseLLMService:
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    async def _forward_request(self, target_url: str, request: Request) -> Response:
        logger.info(f"Forwarding request to: {target_url}")
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=dict(request.headers),
                content=await request.body(),
                timeout=self.timeout,
            )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )