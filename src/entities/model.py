import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Model(BaseModel):
    id: str = Field(min_length=1)
    repo: str
    variant: str | None = None
    backend: Literal["llama.cpp", "ollama"]

    @field_validator("repo")
    @classmethod
    def validate_repo(cls, v):
        if not re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$", v):
            raise ValueError("repo must be in format user/repo")
        return v
