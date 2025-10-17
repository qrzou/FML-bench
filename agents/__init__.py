"""
Agents package for AI research agents.
"""
from .base import BaseAgent, AgentConfig, AgentState, AgentType
from .theaiscientist.theaiscientist import TheAIScientistAgent
from .registry import AgentRegistry

__all__ = [
    "BaseAgent",
    "AgentConfig", 
    "AgentState",
    "AgentType",
    "TheAIScientistAgent",
    "AgentRegistry"
]