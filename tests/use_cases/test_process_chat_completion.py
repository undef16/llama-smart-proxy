from unittest.mock import AsyncMock, MagicMock

import pytest

from src.use_cases.process_chat_completion import ProcessChatCompletion


class TestProcessChatCompletion:
    @pytest.fixture
    def mock_llm_service(self):
        return AsyncMock()

    @pytest.fixture
    def mock_agent_manager(self):
        return MagicMock()

    @pytest.fixture
    def use_case(self, mock_llm_service, mock_agent_manager):
        return ProcessChatCompletion(mock_llm_service, mock_agent_manager)

    @pytest.mark.asyncio
    async def test_execute_processes_request_with_hooks_and_calls_llm(
        self, use_case, mock_llm_service, mock_agent_manager,
    ):
        # Arrange
        request = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}
        processed_request = {"model": "test-model", "messages": [{"role": "user", "content": "Processed Hello"}]}
        llm_response = {"choices": [{"message": {"content": "Response"}}]}
        final_response = {"choices": [{"message": {"content": "Final Response"}}]}

        mock_agent_manager.parse_slash_commands.return_value = []
        mock_agent_manager.build_agent_chain.return_value = []
        mock_agent_manager.execute_request_hooks.return_value = processed_request
        mock_llm_service.generate_completion.return_value = llm_response
        mock_agent_manager.execute_response_hooks.return_value = final_response

        # Act
        result = await use_case.execute(request)

        # Assert
        mock_agent_manager.parse_slash_commands.assert_called_once_with("Hello")
        mock_agent_manager.build_agent_chain.assert_called_once_with([])
        mock_agent_manager.execute_request_hooks.assert_called_once_with(request, [])
        mock_llm_service.generate_completion.assert_called_once_with(processed_request)
        mock_agent_manager.execute_response_hooks.assert_called_once_with(llm_response, [])
        assert result == final_response

    @pytest.mark.asyncio
    async def test_execute_with_agent_chain(self, use_case, mock_llm_service, mock_agent_manager):
        # Arrange
        request = {"model": "test-model", "messages": [{"role": "user", "content": "/agent1 Hello"}]}
        agent_chain = ["agent1"]
        processed_request = {"model": "test-model", "messages": [{"role": "user", "content": "Processed Hello"}]}
        llm_response = {"choices": [{"message": {"content": "Response"}}]}
        final_response = {"choices": [{"message": {"content": "Final Response"}}]}

        mock_agent_manager.parse_slash_commands.return_value = ["agent1"]
        mock_agent_manager.build_agent_chain.return_value = agent_chain
        mock_agent_manager.execute_request_hooks.return_value = processed_request
        mock_llm_service.generate_completion.return_value = llm_response
        mock_agent_manager.execute_response_hooks.return_value = final_response

        # Act
        result = await use_case.execute(request)

        # Assert
        mock_agent_manager.parse_slash_commands.assert_called_once_with("/agent1 Hello")
        mock_agent_manager.build_agent_chain.assert_called_once_with(["agent1"])
        mock_agent_manager.execute_request_hooks.assert_called_once_with(request, agent_chain)
        mock_llm_service.generate_completion.assert_called_once_with(processed_request)
        mock_agent_manager.execute_response_hooks.assert_called_once_with(llm_response, agent_chain)
        assert result == final_response
