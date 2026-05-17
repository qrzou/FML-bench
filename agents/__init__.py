"""
Agents package for AI research agents.
"""
from .base import BaseAgent, AgentConfig, AgentType, AgentResult, StepResult
from .theaiscientist.theaiscientist import TheAIScientistAgent
from .registry import AgentRegistry
from .ai_scientist_v2.agent import AIScientistV2Agent
from .aide.agent import AIDEAgent
from .aira_mcts.agent import AIRAMCTSAgent
from .openevolve.agent import OpenEvolveAgent
from .autoresearch.agent import AutoresearchAgent
from .adaptivesearch.agent import AdaptiveSearchAgent

__all__ = [
    "BaseAgent",
    "AgentConfig",
    "AgentType",
    "AgentResult",
    "StepResult",
    "TheAIScientistAgent",
    "AgentRegistry",
    "AIScientistV2Agent",
    "AIDEAgent",
    "AIRAMCTSAgent",
    "OpenEvolveAgent",
    "AutoresearchAgent",
    "AdaptiveSearchAgent",
]
