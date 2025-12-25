
from pydantic import BaseModel, Field, field_validator


class Agent(BaseModel):
    name: str = Field(min_length=1)
    enabled: bool
    hooks: list[str]

    @field_validator("hooks")
    @classmethod
    def validate_hooks(cls, v):
        valid_hooks = {"request", "response"}
        if not all(hook in valid_hooks for hook in v):
            raise ValueError(f"Invalid hooks. Valid hooks are: {valid_hooks}")
        return v
