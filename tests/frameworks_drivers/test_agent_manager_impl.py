from unittest.mock import MagicMock

import pytest

from src.frameworks_drivers.agent_manager import AgentManager


class TestAgentManager:
    @pytest.fixture
    def agent_manager(self):
        return AgentManager()

    def test_parse_slash_commands_returns_list_of_agent_names_from_content(self, agent_manager):
        # Arrange
        content = "/agent1 Hello /agent2 world"
        agent_manager.agents = {"agent1": MagicMock(), "agent2": MagicMock()}

        # Act
        result = agent_manager.parse_slash_commands(content)

        # Assert
        assert result == ["agent1", "agent2"]

    def test_parse_slash_commands_returns_empty_list_when_no_commands(self, agent_manager):
        # Arrange
        content = "Hello world"

        # Act
        result = agent_manager.parse_slash_commands(content)

        # Assert
        assert result == []

    def test_parse_slash_commands_ignores_unknown_agents(self, agent_manager):
        # Arrange
        content = "/agent1 /unknown Hello"
        agent_manager.agents = {"agent1": MagicMock()}

        # Act
        result = agent_manager.parse_slash_commands(content)

        # Assert
        assert result == ["agent1"]

    def test_build_agent_chain_returns_list_of_agents_for_names(self, agent_manager):
        # Arrange
        agent_names = ["agent1", "agent2"]
        # Mock agents
        agent_manager.agents = {"agent1": MagicMock(), "agent2": MagicMock()}

        # Act
        result = agent_manager.build_agent_chain(agent_names)

        # Assert
        assert len(result) == 2
        assert result[0] == agent_manager.agents["agent1"]
        assert result[1] == agent_manager.agents["agent2"]

    def test_build_agent_chain_ignores_unknown_agent_names(self, agent_manager):
        # Arrange
        agent_names = ["agent1", "unknown"]
        agent_manager.agents = {"agent1": MagicMock()}

        # Act
        result = agent_manager.build_agent_chain(agent_names)

        # Assert
        assert len(result) == 1
        assert result[0] == agent_manager.agents["agent1"]

    def test_execute_request_hooks_calls_process_request_on_each_agent(self, agent_manager):
        # Arrange
        request = {"model": "test"}
        agent_chain = [MagicMock(), MagicMock()]
        processed_request = {"model": "test", "processed": True}

        agent_chain[0].process_request.return_value = {"model": "test", "step1": True}
        agent_chain[1].process_request.return_value = processed_request

        # Act
        result = agent_manager.execute_request_hooks(request, agent_chain)

        # Assert
        agent_chain[0].process_request.assert_called_once_with(request)
        agent_chain[1].process_request.assert_called_once_with({"model": "test", "step1": True})
        assert result == processed_request

    def test_execute_response_hooks_calls_process_response_on_each_agent(self, agent_manager):
        # Arrange
        response = {"choices": [{"message": "test"}]}
        agent_chain = [MagicMock(), MagicMock()]
        processed_response = {"choices": [{"message": "test", "processed": True}]}

        agent_chain[0].process_response.return_value = {"choices": [{"message": "test", "step1": True}]}
        agent_chain[1].process_response.return_value = processed_response

        # Act
        result = agent_manager.execute_response_hooks(response, agent_chain)

        # Assert
        agent_chain[0].process_response.assert_called_once_with(response)
        agent_chain[1].process_response.assert_called_once_with({"choices": [{"message": "test", "step1": True}]})
        assert result == processed_response

    def test_execute_hooks_handles_exceptions_gracefully(self, agent_manager):
        # Arrange
        request = {"model": "test"}
        agent_chain = [MagicMock(), MagicMock()]

        agent_chain[0].process_request.side_effect = Exception("Error")
        agent_chain[1].process_request.return_value = {"model": "test", "fallback": True}

        # Act
        result = agent_manager.execute_request_hooks(request, agent_chain)

        # Assert
        agent_chain[0].process_request.assert_called_once_with(request)
        agent_chain[1].process_request.assert_called_once_with(request)  # Original request since first failed
        assert result == {"model": "test", "fallback": True}
