
from src.shared.protocols import AgentManagerProtocol, LLMServiceProtocol


class ProcessChatCompletion:
    def __init__(self, llm_service: LLMServiceProtocol, agent_manager: AgentManagerProtocol):
        self.llm_service = llm_service
        self.agent_manager = agent_manager

    async def execute(self, request: dict) -> dict:
        # Extract the last user message content for agent parsing
        messages = request.get("messages", [])
        content = ""
        if messages and messages[-1].get("role") == "user":
            content = messages[-1].get("content", "")

        # Parse slash commands to build agent chain
        agent_names = self.agent_manager.parse_slash_commands(content)
        agent_chain = self.agent_manager.build_agent_chain(agent_names)

        # Execute request hooks
        processed_request = self.agent_manager.execute_request_hooks(request, agent_chain)

        # Generate completion using LLM service
        llm_response = await self.llm_service.generate_completion(processed_request)

        # Execute response hooks
        final_response = self.agent_manager.execute_response_hooks(llm_response, agent_chain)

        return final_response
