import time
import uuid
import asyncio
import json
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
import httpx

from .config import Config
from .server_pool import ServerPool
from .agent_manager import AgentManager
from .types import Message
from .common_imports import Logger

logger = Logger.get(__name__)


class API:
    """
    Main API class for the Llama Smart Proxy.
    Encapsulates FastAPI app and all handlers with proper OOP design.
    """

    def __init__(self, config: Config):
        """
        Initialize the API with configuration.

        Args:
            config: Configuration object containing server pool, models, agents, and backend settings.
        """
        self.config = config

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Handle application startup and shutdown events."""
            # Startup logic can be added here if needed
            yield
            # Shutdown logic
            self.server_pool.shutdown()

        self.app = FastAPI(title="Llama Smart Proxy", version="0.1.0", lifespan=lifespan)
        self.server_pool = ServerPool(config.server_pool)
        self.agent_manager = AgentManager(enabled_agents=config.agents)

        # Register routes
        self._register_routes()

    async def _forward_request(self, request: Request, path: str) -> Response:
        """
        Handle forwarding endpoints using the server pool.
        """
        try:
            body = await request.json()
            model = body.get("model")
            if not model:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Model not specified"}
                )

            logger.debug(f"Getting server for model: {model}")
            server = await self.server_pool.get_server_for_model(model)
            if server is None or server.process is None:
                return JSONResponse(
                    status_code=503,
                    content={"error": "No available server for the requested model"}
                )
                
            url = f"http://localhost:{server.port}/{path}"
            logger.debug(f"Forwarding to {url}")

            async with httpx.AsyncClient() as client:
                if request.method == "GET":
                    response = await client.get(url, timeout=30.0)
                elif request.method == "POST":
                    response = await client.post(url, json=body, timeout=30.0)
                else:
                    return JSONResponse(status_code=405, content={"error": "Method not allowed"})

                return JSONResponse(status_code=response.status_code, content=response.json())

        except Exception as e:
            logger.error(f"Error handling forwarding request: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal proxy error", "detail": str(e)}
            )

    def _register_routes(self):
        """
        Register all routes supported by llama.cpp server.

        Custom routes (/chat/completions, /completions, /health) are handled with full processing
        including agent execution and model resolution. All other endpoints are forwarded to the
        backend llama.cpp server without any processing or modification.
        """
        # Custom routes with processing
        self.app.post("/chat/completions")(self.chat_completions)
        self.app.post("/completions")(self.chat_completions)
        self.app.get("/health")(self.health)

        # Catch-all route for forwarding all other requests to backend server without processing
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])(self._forward_request)

    async def _generate_completion(self, server, request: dict) -> dict:
        """
        Generate completion by forwarding to llama-server subprocess.
        """
        try:
            # Prepare the request data for llama-server OpenAI API

            url = f"http://localhost:{server.port}/v1/chat/completions"
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=request, timeout=60.0)
                if response.status_code != 200:
                    raise Exception(f"llama-server returned {response.status_code}: {response.text}")

                result = response.json()
                return result
        except Exception as e:
            logger.debug(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model generation failed: {str(e)}")

    async def chat_completions(self, request: dict) -> dict:
        """
        Handle chat completion requests with model resolution, server selection,
        agent execution, and llama.cpp forwarding.
        """
        try:
            # Extract the last user message for agent parsing

            # Parse slash commands to build agent chain
            agent_names = self.agent_manager.parse_slash_commands(request.get("content", ""))
            agent_chain = self.agent_manager.build_agent_chain(agent_names)

            # Execute request hooks
            processed_request = self.agent_manager.execute_request_hooks(request, agent_chain)

            logger.debug("Getting server for model")
            # Get server for the model
            model = processed_request.get("model")
            if not model or not isinstance(model, str):
                raise HTTPException(status_code=400, detail="Model must be a non-empty string")
            
            server = await self.server_pool.get_server_for_model(model)
            logger.debug(f"Server obtained: {server is not None}")
            if server is None or server.process is None:
                raise HTTPException(status_code=503, detail="No available server for the requested model")

            # Generate completion by forwarding to llama-server
            processed_response = await self._generate_completion(
                server,
                processed_request
            )


            # Execute response hooks
            final_response = self.agent_manager.execute_response_hooks(processed_response, agent_chain)

            return final_response

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    async def health(self):
        """
        Return the health status of the server pool.
        """
        try:
            # Update health status
            await self.server_pool.check_health()
            status = self.server_pool.get_pool_status()
            return JSONResponse(content=status)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

