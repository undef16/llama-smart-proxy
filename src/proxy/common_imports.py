import logging
from typing import Dict, List

from pydantic import BaseModel, Field


class Logger:
    @staticmethod
    def get(name: str) -> logging.Logger:
        """
        Get a standardized logger for the application.
        Configures logging with basic setup if not already configured.
        """
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        return logging.getLogger(name)