"""
Agent registry for managing different agent types.
"""
from typing import Dict, Type, Optional

from .base import BaseAgent, AgentType
from .theaiscientist.theaiscientist import TheAIScientistAgent


class AgentRegistry:
    """Registry for managing agent implementations"""
    
    _agents: Dict[AgentType, Type[BaseAgent]] = {}
    
    @classmethod
    def register(cls, agent_type: AgentType, agent_class: Type[BaseAgent]):
        """
        Register an agent implementation.
        
        Args:
            agent_type: The type of agent
            agent_class: The agent class implementation
        """
        cls._agents[agent_type] = agent_class
    
    @classmethod
    def get(cls, agent_type: AgentType) -> Optional[Type[BaseAgent]]:
        """
        Get an agent class by type.
        
        Args:
            agent_type: The type of agent
            
        Returns:
            The agent class or None if not found
        """
        return cls._agents.get(agent_type)
    
    @classmethod
    def create(cls, agent_type: AgentType, config) -> BaseAgent:
        """
        Create an agent instance.
        
        Args:
            agent_type: The type of agent
            config: Agent configuration
            
        Returns:
            Agent instance
            
        Raises:
            ValueError: If agent type not found
        """
        agent_class = cls.get(agent_type)
        if not agent_class:
            raise ValueError(f"Agent type {agent_type} not registered")
        
        return agent_class(config)
    
    @classmethod
    def list_agents(cls) -> list:
        """
        List all registered agent types.
        
        Returns:
            List of agent types
        """
        return list(cls._agents.keys())


# Register built-in agents
AgentRegistry.register(AgentType.THEAISCIENTIST, TheAIScientistAgent)

# Placeholder registrations for future agents
# e.g.
# AgentRegistry.register(AgentType.XXXXX, XXXXXAgent)