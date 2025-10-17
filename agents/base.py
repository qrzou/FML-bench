"""
Base agent abstraction for AI research agents.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional


class AgentType(Enum):
    """Supported agent types"""
    THEAISCIENTIST = "theaiscientist"


@dataclass
class AgentConfig:
    """Configuration for agent initialization"""
    agent_type: AgentType
    model: str = "gpt-4"
    provider: str = "OpenAI"
    agent_params: Dict[str, Any] = field(default_factory=dict)  # Agent-specific params from config
    runtime_params: Dict[str, Any] = field(default_factory=dict)  # Runtime params (repo_dir, etc.)


@dataclass
class AgentState:
    """Generic state container for different agent types"""
    # iteration: int = 0
    completed: bool = False
    # Agent-specific state stored here
    data: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all AI research agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.state = AgentState()
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the agent with configuration"""
        pass
    
    @abstractmethod
    def run(self, prompt: str) -> str:
        """
        Main interaction method - process prompt and return response.
        This is where the agent does its main work.
        
        Args:
            prompt: The task prompt or instruction
            
        Returns:
            Response string from the agent
        """
        pass
    
    def set_files(self, files: List[str]) -> None:
        """
        Deprecated: target files should be passed to run() method.
        This method is kept for backward compatibility.
        """
        pass
    
    def get_state(self) -> AgentState:
        """
        Get current agent state.
        
        Returns:
            Current AgentState
        """
        return self.state
    
    def cleanup(self) -> None:
        """
        Clean up agent resources. Override if needed.
        """
        pass
    
