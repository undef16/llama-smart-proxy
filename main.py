import uvicorn
from src.proxy.api import API
from src.proxy.config import Config
from src.proxy.common_imports import Logger

if __name__ == "__main__":
    logger = Logger.get(__name__)

    try:
        config = Config.load()
        api = API(config)
        logger.info("Starting Llama Smart Proxy...")
        # Start the uvicorn server
        uvicorn.run(api.app, host=config.server.host, port=config.server.port)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise