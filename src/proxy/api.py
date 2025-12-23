import time
import uuid
import asyncio
import json
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
import requests

from .config import Config
from .server_pool import ServerPool
from .agent_manager import AgentManager
from .types import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatCompletionUsage, Message
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

        self.app = FastAPI(title="Llama Smart Proxy", version="1.0.0", lifespan=lifespan)
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
            if server is None or server.llama is None:
                return JSONResponse(
                    status_code=503,
                    content={"error": "No available server for the requested model"}
                )

            path = request.url.path
            logger.debug(f"Handling endpoint: {path}")

            if path == "/tokenize":
                content = body.get("content", "")
                tokens = await asyncio.to_thread(
                    server.llama.tokenize,
                    content.encode('utf-8')
                )
                return JSONResponse({"tokens": tokens})

            elif path == "/detokenize":
                tokens = body.get("tokens", [])
                content = await asyncio.to_thread(
                    server.llama.detokenize,
                    tokens
                )
                return JSONResponse({"content": content.decode('utf-8')})

            elif path == "/embedding":
                content = body.get("content", "")
                embedding = await asyncio.to_thread(
                    server.llama.embed,
                    content
                )
                return JSONResponse({"embedding": embedding})

            elif path == "/props":
                props = {
                    "model": server.model,
                    "vocab_size": server.llama.n_vocab(),
                    "context_length": server.llama.n_ctx(),
                    "embedding_size": server.llama.n_embd(),
                    "eos_token": server.llama.token_eos(),
                    "bos_token": server.llama.token_bos(),
                    "nl_token": server.llama.token_nl(),
                    "pooling_type": server.llama.pooling_type(),
                    "metadata": server.llama.metadata,
                }
                return JSONResponse(props)

            else:
                return JSONResponse(
                    status_code=404,
                    content={"error": "Endpoint not found"}
                )

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

    async def _generate_completion(self, server, messages: List[Message]) -> str:
        """
        Generate completion using llama.cpp server with proper error handling.
        """
        logger.debug(f"Starting generation with messages: {[msg.dict() for msg in messages]}")
        try:
            # Format messages into a chat prompt string
            prompt_parts = []
            for msg in messages:
                if msg.role == "system":
                    prompt_parts.append(f"System: {msg.content}")
                elif msg.role == "user":
                    prompt_parts.append(f"User: {msg.content}")
                elif msg.role == "assistant":
                    prompt_parts.append(f"Assistant: {msg.content}")
            prompt = "\n".join(prompt_parts) + "\nAssistant:"
            logger.debug(f"Formatted prompt: {prompt}")

            output = await asyncio.to_thread(
                server.llama,
                prompt,
                max_tokens=100,  # Limit response length
                stop=["\n"],  # Basic stop sequence
            )
            logger.debug(f"Generation completed, output: {output}")
            return output["choices"][0]["text"].strip()
        except Exception as e:
            logger.debug(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model generation failed: {str(e)}")

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Handle chat completion requests with model resolution, server selection,
        agent execution, and llama.cpp forwarding.
        """
        try:
            # Extract the last user message for agent parsing
            if not request.messages:
                raise HTTPException(status_code=400, detail="No messages provided")

            last_message = request.messages[-1]
            if last_message.role != "user":
                raise HTTPException(status_code=400, detail="Last message must be from user")

            # Parse slash commands to build agent chain
            agent_names = self.agent_manager.parse_slash_commands(last_message.content)
            agent_chain = self.agent_manager.build_agent_chain(agent_names)

            # Execute request hooks
            processed_request = self.agent_manager.execute_request_hooks(request, agent_chain)

            logger.debug("Getting server for model")
            # Get server for the model
            server = await self.server_pool.get_server_for_model(processed_request.model)
            logger.debug(f"Server obtained: {server is not None}")
            if server is None or server.llama is None:
                raise HTTPException(status_code=503, detail="No available server for the requested model")

            # Generate completion using llama.cpp
            completion_text = await self._generate_completion(
                server,
                processed_request.messages
            )

            # Create response
            response_id = str(uuid.uuid4())
            created = int(time.time())

            choice = ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=completion_text),
                finish_reason="stop"
            )

            # Estimate usage (llama.cpp doesn't provide exact token counts)
            prompt_tokens = sum(len(msg.content.split()) for msg in processed_request.messages)  # Rough estimate
            completion_tokens = len(completion_text.split())  # Rough estimate

            usage = ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )

            response = ChatCompletionResponse(
                id=response_id,
                object="chat.completion",
                created=created,
                model=processed_request.model,
                choices=[choice],
                usage=usage
            )

            # Execute response hooks
            final_response = self.agent_manager.execute_response_hooks(response, agent_chain)

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

