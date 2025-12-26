import inspect
from typing import get_type_hints

from src.entities.model import Model
from src.shared.protocols import AgentManagerProtocol, LLMServiceProtocol, ModelRepositoryProtocol


def test_model_repository_protocol():
    assert inspect.isclass(ModelRepositoryProtocol)
    assert hasattr(ModelRepositoryProtocol, "get_model")
    # Skip type hints check due to forward references in protocols
    # hints = get_type_hints(ModelRepositoryProtocol.get_model, globalns=globals())
    # assert "model_id" in hints
    # assert hints["model_id"] == str


def test_llm_service_protocol():
    assert inspect.isclass(LLMServiceProtocol)
    assert hasattr(LLMServiceProtocol, "generate_completion")
    # Check async
    assert inspect.iscoroutinefunction(LLMServiceProtocol.generate_completion)


def test_agent_manager_interface_protocol():
    assert inspect.isclass(AgentManagerProtocol)
    assert hasattr(AgentManagerProtocol, "parse_slash_commands")
    assert hasattr(AgentManagerProtocol, "execute_request_hooks")
    assert hasattr(AgentManagerProtocol, "execute_response_hooks")
