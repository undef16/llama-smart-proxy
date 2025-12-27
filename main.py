import os
import uvicorn

from src.frameworks_drivers.agent_manager import AgentManager
from src.frameworks_drivers.config import Config
from src.frameworks_drivers.gpu.gpu_monitor import GPUMonitor
from src.frameworks_drivers.llm_service_factory import LLMServiceFactory
from src.frameworks_drivers.model_repository import ModelRepository
from src.frameworks_drivers.model_resolver import ModelResolver
from src.frameworks_drivers.server_pool import ServerPool
from src.interface_adapters.api import API
from src.interface_adapters.chat_controller import ChatController
from src.interface_adapters.health_controller import HealthController
from src.shared.logger import Logger
from src.use_cases.get_health import GetHealth
from src.use_cases.process_chat_completion import ProcessChatCompletion

if __name__ == "__main__":
    logger = Logger.get(__name__)

    try:
        config = Config.load()

        # Override backend if set in environment
        if "LLM_PROXY_BACKEND" in os.environ:
            config.backend = os.environ["LLM_PROXY_BACKEND"]

        # Instantiate dependencies
        model_repository = ModelRepository()
        agent_manager = AgentManager()
        server_pool = ServerPool(config.server_pool, model_repository, full_config=config)
        model_resolver = ModelResolver()
        gpu_monitor = GPUMonitor(config)
        llm_service_factory = LLMServiceFactory(config.model_dump(), server_pool, model_resolver)
        llm_service = llm_service_factory.create_service()

        # Instantiate use cases
        process_chat_completion = ProcessChatCompletion(llm_service, agent_manager)
        get_health = GetHealth(model_repository, gpu_monitor, config)

        # Instantiate controllers
        chat_controller = ChatController(process_chat_completion)
        health_controller = HealthController(get_health)

        # Instantiate API
        api = API(chat_controller, health_controller)

        logger.info("Starting Llama Smart Proxy...")
        # Start the uvicorn server
        uvicorn.run(api.app, host=config.server.host, port=config.server.port)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
