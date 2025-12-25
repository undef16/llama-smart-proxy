"""
Unit tests for agent_manager.py module.
"""

import json
from unittest.mock import Mock, patch

from src.frameworks_drivers.agent_manager import AgentManager


class TestAgentManager:
    """Test AgentManager class."""

    def test_init_with_no_plugins_dir(self, temp_dir):
        """Test initialization when plugins directory doesn't exist."""
        nonexistent_dir = temp_dir / "nonexistent_plugins"
        manager = AgentManager(str(nonexistent_dir))
        assert manager.agents == {}
        assert manager.agent_configs == {}
        assert manager.enabled_agents == []

    def test_init_with_enabled_agents(self, mock_plugins_dir):
        """Test initialization with specific enabled agents."""
        enabled_agents = ["test_agent"]
        manager = AgentManager(mock_plugins_dir, enabled_agents)
        assert "test_agent" in manager.agents
        assert "another_agent" not in manager.agents
        assert manager.enabled_agents == enabled_agents

    def test_discover_and_load_agents(self, mock_plugins_dir):
        """Test discovering and loading agents from plugins directory."""
        manager = AgentManager(mock_plugins_dir)
        assert "test_agent" in manager.agents
        assert "another_agent" in manager.agents
        assert len(manager.agents) == 2

        # Check agent configs
        assert "test_agent" in manager.agent_configs
        assert "another_agent" in manager.agent_configs
        assert manager.agent_configs["test_agent"]["enabled"] is True

    def test_discover_disabled_agents(self, temp_dir, mock_agent_config):
        """Test that disabled agents are not loaded."""
        plugins_dir = temp_dir / "plugins"
        plugins_dir.mkdir()

        agent_dir = plugins_dir / "disabled_agent"
        agent_dir.mkdir()

        # Create config with disabled agent
        config_path = agent_dir / "config.json"
        disabled_config = mock_agent_config.model_dump()
        disabled_config["enabled"] = False
        with open(config_path, "w") as f:
            json.dump(disabled_config, f, indent=2)

        # Create agent.py
        agent_py = agent_dir / "agent.py"
        with open(agent_py, "w") as f:
            f.write(
                """
class DisabledAgent:
    def __init__(self):
        self.name = "disabled_agent"

    def process_request(self, request):
        return request

    def process_response(self, response):
        return response
""",
            )

        manager = AgentManager(str(plugins_dir))
        assert "disabled_agent" not in manager.agents
        assert len(manager.agents) == 0

    def test_discover_incomplete_agents(self, incomplete_plugins_dir):
        """Test handling of incomplete agent setups."""
        manager = AgentManager(incomplete_plugins_dir)
        assert "incomplete_agent" not in manager.agents
        assert len(manager.agents) == 0

    def test_parse_slash_commands_no_commands(self, sample_chat_request):
        """Test parsing slash commands when none are present."""
        manager = AgentManager()
        commands = manager.parse_slash_commands("Hello, how are you?")
        assert commands == []

    def test_parse_slash_commands_single_command(self, sample_chat_request):
        """Test parsing a single slash command."""
        manager = AgentManager()
        # Mock agents
        manager.agents = {"test": Mock(), "another": Mock()}

        commands = manager.parse_slash_commands("/test Hello world")
        assert commands == ["test"]

    def test_parse_slash_commands_multiple_commands(self, sample_chat_request):
        """Test parsing multiple slash commands."""
        manager = AgentManager()
        manager.agents = {"test": Mock(), "another": Mock(), "unknown": Mock()}

        commands = manager.parse_slash_commands("/test /another /unknown_command hello")
        assert commands == ["test", "another"]  # unknown_command should be ignored

    def test_parse_slash_commands_mixed_text(self, sample_chat_request):
        """Test parsing slash commands mixed with regular text."""
        manager = AgentManager()
        manager.agents = {"test": Mock()}

        commands = manager.parse_slash_commands("Please /test process this text /unknown")
        assert commands == ["test"]

    def test_parse_slash_commands_empty_string(self):
        """Test parsing slash commands from empty string."""
        manager = AgentManager()
        commands = manager.parse_slash_commands("")
        assert commands == []

    def test_parse_slash_commands_only_slash(self):
        """Test parsing slash commands with only slash."""
        manager = AgentManager()
        commands = manager.parse_slash_commands("/")
        assert commands == []

    def test_build_agent_chain(self, mock_plugins_dir):
        """Test building agent chain in specified order."""
        manager = AgentManager(mock_plugins_dir)
        chain = manager.build_agent_chain(["test_agent", "another_agent"])
        assert len(chain) == 2
        assert hasattr(chain[0], "process_request")  # Should be agent instances

    def test_build_agent_chain_unknown_agents(self, mock_plugins_dir):
        """Test building agent chain with unknown agents."""
        manager = AgentManager(mock_plugins_dir)
        chain = manager.build_agent_chain(["test_agent", "unknown_agent", "another_agent"])
        assert len(chain) == 2  # unknown_agent should be skipped

    def test_build_agent_chain_empty_list(self, mock_plugins_dir):
        """Test building agent chain with empty list."""
        manager = AgentManager(mock_plugins_dir)
        chain = manager.build_agent_chain([])
        assert chain == []

    def test_execute_request_hooks(self, mock_plugins_dir, sample_chat_request):
        """Test executing request hooks in order."""
        manager = AgentManager(mock_plugins_dir)

        # Mock agent methods
        for agent in manager.agents.values():
            agent.process_request = Mock(return_value=sample_chat_request)

        chain = manager.build_agent_chain(["test_agent", "another_agent"])
        result = manager.execute_request_hooks(sample_chat_request, chain)

        assert result == sample_chat_request
        # Verify each agent was called once
        for agent in chain:
            agent.process_request.assert_called_once_with(sample_chat_request)

    def test_execute_request_hooks_with_exception(self, mock_plugins_dir, sample_chat_request):
        """Test executing request hooks when one agent raises exception."""
        manager = AgentManager(mock_plugins_dir)

        # Mock agents - one raises exception
        agents = list(manager.agents.values())
        agents[0].process_request = Mock(side_effect=Exception("Test error"))
        agents[1].process_request = Mock(return_value=sample_chat_request)

        chain = [agents[0], agents[1]]
        result = manager.execute_request_hooks(sample_chat_request, chain)

        # Should continue with other agents despite exception
        assert result == sample_chat_request
        agents[0].process_request.assert_called_once()
        agents[1].process_request.assert_called_once()

    def test_execute_response_hooks(self, mock_plugins_dir, sample_chat_response):
        """Test executing response hooks in order."""
        manager = AgentManager(mock_plugins_dir)

        # Mock agent methods
        for agent in manager.agents.values():
            agent.process_response = Mock(return_value=sample_chat_response)

        chain = manager.build_agent_chain(["test_agent", "another_agent"])
        result = manager.execute_response_hooks(sample_chat_response, chain)

        assert result == sample_chat_response
        # Verify each agent was called once
        for agent in chain:
            agent.process_response.assert_called_once_with(sample_chat_response)

    def test_execute_response_hooks_with_exception(self, mock_plugins_dir, sample_chat_response):
        """Test executing response hooks when one agent raises exception."""
        manager = AgentManager(mock_plugins_dir)

        # Mock agents - one raises exception
        agents = list(manager.agents.values())
        agents[0].process_response = Mock(side_effect=Exception("Test error"))
        agents[1].process_response = Mock(return_value=sample_chat_response)

        chain = [agents[0], agents[1]]
        result = manager.execute_response_hooks(sample_chat_response, chain)

        # Should continue with other agents despite exception
        assert result == sample_chat_response
        agents[0].process_response.assert_called_once()
        agents[1].process_response.assert_called_once()

    def test_get_available_agents(self, mock_plugins_dir):
        """Test getting list of available agents."""
        manager = AgentManager(mock_plugins_dir)
        available = manager.get_available_agents()
        assert set(available) == {"test_agent", "another_agent"}

    def test_get_available_agents_empty(self):
        """Test getting available agents when none are loaded."""
        manager = AgentManager()
        available = manager.get_available_agents()
        assert available == []

    @patch("importlib.util.spec_from_file_location")
    @patch("importlib.util.module_from_spec")
    def test_agent_loading_error_handling(self, mock_module_from_spec, mock_spec_from_file, temp_dir):
        """Test error handling during agent loading."""
        # Create a plugins directory with a problematic agent
        plugins_dir = temp_dir / "plugins"
        plugins_dir.mkdir()
        agent_dir = plugins_dir / "problematic_agent"
        agent_dir.mkdir()

        # Create config
        config_path = agent_dir / "config.json"
        with open(config_path, "w") as f:
            f.write('{"description": "test", "enabled": true, "end_point": "/chat/completions"}')

        # Create agent.py that will cause import error
        agent_py = agent_dir / "agent.py"
        with open(agent_py, "w") as f:
            f.write("raise ImportError('Test import error')")

        # Mock spec to return None (simulating import failure)
        mock_spec_from_file.return_value = None

        manager = AgentManager(str(plugins_dir))

        # Should not crash, agent should not be loaded
        assert "problematic_agent" not in manager.agents
