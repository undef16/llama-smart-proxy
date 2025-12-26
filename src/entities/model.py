import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Model(BaseModel):
    id: str = Field(min_length=1)
    repo: str
    variant: str | None = None
    backend: Literal["llama.cpp", "ollama"]
    estimated_vram: Optional[float] = None  # Estimated VRAM requirement in GB
    parameters: Optional[int] = None  # Number of parameters (e.g., 7B, 13B)

    @field_validator("repo")
    @classmethod
    def validate_repo(cls, v):
        # Allow HuggingFace format (user/repo) or "local" for local files
        # Allow alphanumeric, dots, hyphens, underscores in user and repo names
        if not (re.match(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$", v) or v == "local"):
            raise ValueError("repo must be in format user/repo or 'local'")
        return v
