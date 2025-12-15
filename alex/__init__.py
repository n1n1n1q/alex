"""
Alex - Hierarchical Minecraft Agent

Organized module structure:
- core: Fundamental types, configuration, and state management
- planning: Planners, metaplanner, reflex, and skill routing
- execution: Policy executors and STEVE-1 integration
- prompts: Prompt generation and MCP server
- vision: MineCLIP-based vision system
- utils: Utility functions (serialization, etc.)

Usage:
    from alex import Agent, AgentVision
    from alex.core import get_config, GameState
    from alex.planning import Planner, MetaPlanner
    from alex.execution import execute_policy_skill
"""

__version__ = "0.1.0"

from .agent import Agent, VerboseAgent
from .agent_vision import AgentVision

__all__ = ["Agent", "VerboseAgent", "AgentVision"]
