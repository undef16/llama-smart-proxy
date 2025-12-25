from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Server(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    host: str
    port: int = Field(ge=1, le=65535)
    model_id: str
    status: Literal["stopped", "running", "error"]
    process: int | None = None
