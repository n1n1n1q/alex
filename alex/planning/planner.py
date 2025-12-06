from __future__ import annotations

from typing import List

from ..core.types import GameState, Subgoal
from .base_planner import BasePlanner


class Planner(BasePlanner):
    """Simple rule-based planner.

    Uses basic heuristics for early game progression.
    LLM team: extend this or create a new planner implementation.
    """

    def plan(self, state: GameState) -> List[Subgoal]:
        """
        Generate subgoals using simple rule-based logic.
        
        This is the fallback planner when Gemini is not available.
        """
        return self.fallback_plan(state)
