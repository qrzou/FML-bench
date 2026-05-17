"""
Agent registry for managing different agent types.
"""
from typing import Dict, Type, Optional

from .base import BaseAgent, AgentType
from .theaiscientist.theaiscientist import TheAIScientistAgent
from .ai_scientist_v2.agent import AIScientistV2Agent
from .aide.agent import AIDEAgent
from .aira_mcts.agent import AIRAMCTSAgent
from .openevolve.agent import OpenEvolveAgent
from .autoresearch.agent import AutoresearchAgent
from .adaptivesearch.agent import AdaptiveSearchAgent


class AgentRegistry:
    """Registry for managing agent implementations"""

    _agents: Dict[AgentType, Type[BaseAgent]] = {}

    @classmethod
    def register(cls, agent_type: AgentType, agent_class: Type[BaseAgent]):
        cls._agents[agent_type] = agent_class

    @classmethod
    def get(cls, agent_type: AgentType) -> Optional[Type[BaseAgent]]:
        return cls._agents.get(agent_type)

    @classmethod
    def create(cls, agent_type: AgentType, config) -> BaseAgent:
        agent_class = cls.get(agent_type)
        if not agent_class:
            raise ValueError(f"Agent type {agent_type} not registered")
        return agent_class(config)

    @classmethod
    def list_agents(cls) -> list:
        return list(cls._agents.keys())


AgentRegistry.register(AgentType.THEAISCIENTIST, TheAIScientistAgent)
AgentRegistry.register(AgentType.AI_SCIENTIST_V2, AIScientistV2Agent)
AgentRegistry.register(AgentType.AIDE, AIDEAgent)
AgentRegistry.register(AgentType.AIRA_MCTS, AIRAMCTSAgent)
AgentRegistry.register(AgentType.OPENEVOLVE, OpenEvolveAgent)
AgentRegistry.register(AgentType.AUTORESEARCH, AutoresearchAgent)
AgentRegistry.register(AgentType.ADAPTIVESEARCH, AdaptiveSearchAgent)
