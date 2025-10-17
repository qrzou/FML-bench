"""
TheAIScientist agent module.
"""
from .theaiscientist import TheAIScientistAgent
from .generate_ideas import generate_ideas, check_idea_novelty

__all__ = ["TheAIScientistAgent", "generate_ideas", "check_idea_novelty"]