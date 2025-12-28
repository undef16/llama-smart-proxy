"""
Test configuration and fixtures for llama smart proxy tests.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from src.frameworks_drivers.config import Config, ServerPoolConfig
from src.frameworks_drivers.server_pool import ServerPool


def pytest_configure(config):
    """Configure pytest warnings."""
    config.addinivalue_line("filterwarnings", "ignore::PendingDeprecationWarning")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config_data():
    """Sample configuration data for testing."""
    return {
        "backend": "llama.cpp",
        "server": {"host": "0.0.0.0", "port": 8000},
        "server_pool": {"size": 3, "host": "127.0.0.1", "port_start": 9000, "gpu_layers": 0, "request_timeout": 300},
        "models": {
            "test-model": {"repo": "test/repo", "variant": "test-variant.gguf"},
            "another-model": {"repo": "another/repo", "variant": "Q4_K_M"},
        },
        "agents": ["test-agent", "another-agent"],
    }


@pytest.fixture
def config_file(temp_dir, sample_config_data):
    """Create a temporary config file."""
    config_path = temp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(sample_config_data, f, indent=2)
    return str(config_path)


@pytest.fixture
def sample_config(sample_config_data):
    """Create a Config instance from sample data."""
    return Config(**sample_config_data)


@pytest.fixture
def minimal_config_data():
    """Minimal configuration data with defaults."""
    return {
        "backend": "llama.cpp",
        "server": {"host": "0.0.0.0", "port": 8000},
        "server_pool": {"size": 1, "gpu_layers": 0, "request_timeout": 300},
        "models": {},
        "agents": [],
    }


@pytest.fixture
def invalid_config_data():
    """Invalid configuration data for error testing."""
    return {
        "backend": "llama.cpp",
        "server": {"host": "0.0.0.0", "port": 8000},
        "server_pool": {"size": 0, "host": "localhost", "port_start": 8001, "gpu_layers": 0, "request_timeout": 300},  # Invalid: must be > 0
        "models": {},
        "agents": [],
    }


@pytest.fixture
def model_resolver_test_cases():
    """Test cases for ModelResolver."""
    return [
        # (input, expected_repo, expected_pattern, should_raise)
        ("test/repo", "test/repo", "*.Q4_K_M.gguf", False),
        ("test/repo:Q4_K_M", "test/repo", "*.Q4_K_M.gguf", False),
        ("test/repo:Q4_K_M.gguf", "test/repo", "Q4_K_M.gguf", False),
        ("unsloth/Qwen3-0.6B-GGUF:Q4_K_M", "unsloth/Qwen3-0.6B-GGUF", "*.Q4_K_M.gguf", False),
        ("", None, None, True),
        ("   ", None, None, True),
        ("repo:", None, None, True),
        (":variant", None, None, True),
        ("repo::variant", None, None, True),
    ]


@pytest.fixture
def sample_chat_request():
    """Sample chat completion request for testing."""
    return {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "user", "content": "Can you help me with something?"},
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False,
    }


@pytest.fixture
def sample_chat_response():
    """Sample chat completion response for testing."""
    return {
        "id": "test-response-123",
        "created": 1234567890,
        "model": "test-model",
        "object": "chat.completion",
        "choices": [{"role": "assistant", "content": "I'd be happy to help you!"}],
        "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
    }


@pytest.fixture
def mock_agent():
    """Mock agent for testing."""

    class MockAgent:
        def __init__(self, name: str):
            self.name = name
            self.process_request_calls = []
            self.process_response_calls = []

        def process_request(self, request: dict) -> dict:
            self.process_request_calls.append(request)
            return request

        def process_response(self, response: dict) -> dict:
            self.process_response_calls.append(response)
            return response

    return MockAgent


@pytest.fixture
def mock_agent_config():
    """Mock agent configuration for testing."""

    class MockAgentConfig:
        def __init__(self):
            self.description = "test agent"
            self.enabled = True
            self.end_point = "/chat/completions"

        def model_dump(self):
            return {
                "description": self.description,
                "enabled": self.enabled,
                "end_point": self.end_point,
            }

    return MockAgentConfig()


@pytest.fixture
def mock_plugins_dir(temp_dir):
    """Create a mock plugins directory with test agents."""
    plugins_dir = temp_dir / "plugins"
    plugins_dir.mkdir()

    # Create test_agent
    test_agent_dir = plugins_dir / "test_agent"
    test_agent_dir.mkdir()

    # Create config.json for test_agent
    config_path = test_agent_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "description": "Test agent",
            "enabled": True,
            "end_point": "/chat/completions"
        }, f, indent=2)

    # Create agent.py for test_agent
    agent_py = test_agent_dir / "agent.py"
    with open(agent_py, "w") as f:
        f.write("""
class TestAgent:
    def __init__(self):
        self.name = "test_agent"

    def process_request(self, request):
        return request

    def process_response(self, response):
        return response
""")

    # Create another_agent
    another_agent_dir = plugins_dir / "another_agent"
    another_agent_dir.mkdir()

    # Create config.json for another_agent
    config_path = another_agent_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "description": "Another test agent",
            "enabled": True,
            "end_point": "/chat/completions"
        }, f, indent=2)

    # Create agent.py for another_agent
    agent_py = another_agent_dir / "agent.py"
    with open(agent_py, "w") as f:
        f.write("""
class AnotherAgent:
    def __init__(self):
        self.name = "another_agent"

    def process_request(self, request):
        return request

    def process_response(self, response):
        return response
""")

    return str(plugins_dir)


@pytest.fixture
def incomplete_plugins_dir(temp_dir):
    """Create a mock plugins directory with incomplete agent setup."""
    plugins_dir = temp_dir / "plugins"
    plugins_dir.mkdir()

    # Create incomplete_agent with only config.json (missing agent.py)
    incomplete_agent_dir = plugins_dir / "incomplete_agent"
    incomplete_agent_dir.mkdir()

    # Create config.json for incomplete_agent
    config_path = incomplete_agent_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "description": "Incomplete agent",
            "enabled": True,
            "end_point": "/chat/completions"
        }, f, indent=2)

    # Don't create agent.py - this makes it incomplete

    return str(plugins_dir)


@pytest.fixture
def server_pool_config():
    return ServerPoolConfig(size=2, host="localhost", port_start=8080, gpu_layers=10, request_timeout=300)


@pytest.fixture
def server_pool(server_pool_config):
    return ServerPool(server_pool_config)



