
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Optional

from src.shared.protocols import AgentManagerProtocol


class AgentManager(AgentManagerProtocol):
    def __init__(self, plugins_dir: Optional[str] = None, enabled_agents: Optional[List[str]] = None):
        self.plugins_dir = Path(plugins_dir) if plugins_dir else None
        self.enabled_agents = enabled_agents or []
        self.agents: Dict[str, object] = {}
        self.agent_configs: Dict[str, dict] = {}

        if self.plugins_dir and self.plugins_dir.exists():
            self._discover_and_load_agents()

    def _discover_and_load_agents(self):
        """Discover and load agents from plugins directory."""
        if not self.plugins_dir or not self.plugins_dir.exists():
            return

        for agent_dir in self.plugins_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            agent_name = agent_dir.name
            config_path = agent_dir / "config.json"
            agent_py_path = agent_dir / "agent.py"

            if not config_path.exists() or not agent_py_path.exists():
                continue

            try:
                # Load config
                with open(config_path, 'r') as f:
                    config = json.load(f)

                if not config.get("enabled", False):
                    continue

                # Check if agent is in enabled list (if specified)
                if self.enabled_agents and agent_name not in self.enabled_agents:
                    continue

                # Load agent module
                spec = importlib.util.spec_from_file_location(agent_name, agent_py_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find the agent class (assume it's the only class in the module)
                    agent_class = None
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and hasattr(attr, 'process_request'):
                            agent_class = attr
                            break

                    if agent_class:
                        agent_instance = agent_class()
                        self.agents[agent_name] = agent_instance
                        self.agent_configs[agent_name] = config

            except Exception:
                # Skip problematic agents
                continue

    def parse_slash_commands(self, content: str) -> list[str]:
        """Parse slash commands from content."""
        if not content:
            return []

        parts = content.split()
        commands = []
        for part in parts:
            if part.startswith("/") and len(part) > 1:
                agent_name = part[1:]
                if agent_name in self.agents:
                    commands.append(agent_name)
        return commands

    def build_agent_chain(self, agent_names: list[str]) -> list:
        """Build agent chain for the specified agent names."""
        chain = []
        for name in agent_names:
            if name in self.agents:
                chain.append(self.agents[name])
        return chain

    def execute_request_hooks(self, request: dict, agent_chain: list) -> dict:
        """Execute request hooks on the agent chain."""
        current_obj = request
        for agent in agent_chain:
            try:
                current_obj = agent.process_request(current_obj)
            except Exception:
                # Continue with other agents
                pass
        return current_obj

    def execute_response_hooks(self, response: dict, agent_chain: list) -> dict:
        """Execute response hooks on the agent chain."""
        current_obj = response
        for agent in agent_chain:
            try:
                current_obj = agent.process_response(current_obj)
            except Exception:
                # Continue with other agents
                pass
        return current_obj

    def get_available_agents(self) -> list[str]:
        """Get list of available agent names."""
        return list(self.agents.keys())
