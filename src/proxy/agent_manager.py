import importlib.util
import json
from pathlib import Path
from typing import List, Dict, Optional
from .types import Agent, AgentConfig, ChatCompletionRequest, ChatCompletionResponse
from .common_imports import Logger

logger = Logger.get(__name__)


class AgentManager:
    """Manages discovery, loading, and execution of agent plugins."""

    def __init__(self, plugins_dir: str = "plugins", enabled_agents: Optional[List[str]] = None):
        self.plugins_dir = Path(plugins_dir)
        self.enabled_agents = enabled_agents or []
        self.agents: Dict[str, Agent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self._discover_and_load_agents()

    def _discover_and_load_agents(self):
        """Discover and load agent plugins from the plugins directory."""
        if not self.plugins_dir.exists():
            return

        for agent_dir in self.plugins_dir.iterdir():
            if agent_dir.is_dir():
                agent_name = agent_dir.name
                config_path = agent_dir / "config.json"
                agent_py_path = agent_dir / "agent.py"

                if config_path.exists() and agent_py_path.exists():
                    try:
                        # Load config
                        with open(config_path, 'r') as f:
                            config_data = json.load(f)
                        config = AgentConfig(**config_data)

                        if not config.enabled or (self.enabled_agents and agent_name not in self.enabled_agents):
                            continue

                        # Load agent module
                        spec = importlib.util.spec_from_file_location(agent_name, agent_py_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)

                            # Assume the module has a class named Agent with the same name as the directory
                            # Convert snake_case to CamelCase for class name
                            class_name = ''.join(word.capitalize() for word in agent_name.split('_'))
                            agent_class = getattr(module, class_name, None)
                            if agent_class:
                                agent_instance = agent_class()
                                self.agents[agent_name] = agent_instance
                                self.agent_configs[agent_name] = config
                    except Exception as e:
                        # Log error but continue
                        logger.error(f"Failed to load agent {agent_name}: {e}")

    def parse_slash_commands(self, prompt: str) -> List[str]:
        """Parse slash commands from the prompt and return list of agent names in order."""
        # Split by spaces and filter commands starting with /
        parts = prompt.split()
        commands = []
        for part in parts:
            if part.startswith('/') and len(part) > 1:
                agent_name = part[1:]  # Remove the /
                if agent_name in self.agents:
                    commands.append(agent_name)
                else:
                    # Unknown command, ignore
                    pass
        return commands

    def build_agent_chain(self, agent_names: List[str]) -> List[Agent]:
        """Build a chain of agents in the specified order."""
        chain = []
        for name in agent_names:
            if name in self.agents:
                chain.append(self.agents[name])
        return chain

    def _execute_hooks_safely(self, agent_chain: List[Agent], obj, hook_func):
        """Execute hooks safely with error handling."""
        current_obj = obj
        for agent in agent_chain:
            try:
                current_obj = hook_func(agent, current_obj)
            except Exception as e:
                # Log error but continue with other agents
                logger.error(f"Error in hook for agent {agent}: {e}")
        return current_obj

    def execute_request_hooks(self, request: ChatCompletionRequest, agent_chain: List[Agent]) -> ChatCompletionRequest:
        """Execute request hooks in deterministic order."""
        return self._execute_hooks_safely(agent_chain, request, lambda agent, req: agent.process_request(req))

    def execute_response_hooks(self, response: ChatCompletionResponse, agent_chain: List[Agent]) -> ChatCompletionResponse:
        """Execute response hooks in deterministic order."""
        return self._execute_hooks_safely(agent_chain, response, lambda agent, res: agent.process_response(res))

    def get_available_agents(self) -> List[str]:
        """Return list of available agent names."""
        return list(self.agents.keys())