"""
Test configuration and fixtures for llama smart proxy tests.
"""
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List
import pytest

from src.proxy.config import Config, ServerPoolConfig, ModelConfig
from src.proxy.types import Message, AgentConfig


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
        "server_pool": {
            "size": 3,
            "host": "127.0.0.1",
            "port_start": 9000
        },
        "models": {
            "test-model": {
                "repo": "test/repo",
                "variant": "test-variant.gguf"
            },
            "another-model": {
                "repo": "another/repo",
                "variant": "Q4_K_M"
            }
        },
        "agents": ["test-agent", "another-agent"]
    }


@pytest.fixture
def config_file(temp_dir, sample_config_data):
    """Create a temporary config file."""
    config_path = temp_dir / "config.json"
    with open(config_path, 'w') as f:
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
        "server_pool": {
            "size": 1
        },
        "models": {},
        "agents": []
    }


@pytest.fixture
def invalid_config_data():
    """Invalid configuration data for error testing."""
    return {
        "server_pool": {
            "size": 0,  # Invalid: must be > 0
            "host": "localhost",
            "port_start": 8001
        },
        "models": {},
        "agents": []
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
            {"role": "user", "content": "Can you help me with something?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False
    }


@pytest.fixture
def sample_chat_response():
    """Sample chat completion response for testing."""
    return {
        "id": "test-response-123",
        "created": 1234567890,
        "model": "test-model",
        "object": "chat.completion",
        "choices": [
            {"role": "assistant", "content": "I'd be happy to help you!"}
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 10,
            "total_tokens": 25
        }
    }


@pytest.fixture
def mock_agent_config():
    """Mock agent configuration."""
    return AgentConfig(
        description="A test agent for testing purposes",
        enabled=True,
        end_point="/chat/completions"
    )


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
def mock_plugins_dir(temp_dir, mock_agent_config, mock_agent):
    """Create a mock plugins directory with test agents."""
    plugins_dir = temp_dir / "plugins"
    plugins_dir.mkdir()

    # Create test agent directories
    agent1_dir = plugins_dir / "test_agent"
    agent1_dir.mkdir()
    
    agent2_dir = plugins_dir / "another_agent"
    agent2_dir.mkdir()

    # Create config files for agents
    agent1_config = agent1_dir / "config.json"
    agent2_config = agent2_dir / "config.json"

    with open(agent1_config, 'w') as f:
        json.dump(mock_agent_config.model_dump(), f, indent=2)

    # Different config for second agent
    agent2_config_data = mock_agent_config.model_dump()
    agent2_config_data["description"] = "Another test agent"
    with open(agent2_config, 'w') as f:
        json.dump(agent2_config_data, f, indent=2)

    # Create agent.py files
    agent1_py = agent1_dir / "agent.py"
    agent2_py = agent2_dir / "agent.py"

    # Create simple agent modules
    agent1_content = """
class TestAgent:
    def __init__(self):
        self.name = "test_agent"

    def process_request(self, request):
        return request

    def process_response(self, response):
        return response
"""

    agent2_content = """
class AnotherAgent:
    def __init__(self):
        self.name = "another_agent"

    def process_request(self, request):
        return request

    def process_response(self, response):
        return response
"""

    with open(agent1_py, 'w') as f:
        f.write(agent1_content)

    with open(agent2_py, 'w') as f:
        f.write(agent2_content)

    return str(plugins_dir)


@pytest.fixture
def incomplete_plugins_dir(temp_dir):
    """Create a plugins directory with incomplete agent setup."""
    plugins_dir = temp_dir / "plugins"
    plugins_dir.mkdir()

    # Create agent directory without config.json
    agent_dir = plugins_dir / "incomplete_agent"
    agent_dir.mkdir()
    
    # Create agent.py but no config.json
    agent_py = agent_dir / "agent.py"
    with open(agent_py, 'w') as f:
        f.write("class IncompleteAgent:\n    pass")

    return str(plugins_dir)


@pytest.fixture
def disabled_plugins_dir(temp_dir, mock_agent_config, mock_agent):
    """Create a plugins directory with a disabled agent."""
    plugins_dir = temp_dir / "plugins"
    plugins_dir.mkdir()

    agent_dir = plugins_dir / "disabled_agent"
    agent_dir.mkdir()

    # Create config with disabled agent
    config_path = agent_dir / "config.json"
    disabled_config = mock_agent_config.model_dump()
    disabled_config["enabled"] = False
    with open(config_path, 'w') as f:
        json.dump(disabled_config, f, indent=2)

    # Create agent.py
    agent_py = agent_dir / "agent.py"
    with open(agent_py, 'w') as f:
        f.write("""
class DisabledAgent:
    def __init__(self):
        self.name = "disabled_agent"

    def process_request(self, request):
        return request

    def process_response(self, response):
        return response
""")

