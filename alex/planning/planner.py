from __future__ import annotations

from typing import List

from ..core.types import GameState, Subgoal
from .base_planner import BasePlanner


class Planner(BasePlanner):
    """Simple rule-based planner.

    Uses basic heuristics for early game progression.
    """

    def plan(self, state: GameState) -> List[Subgoal]:
        """
        Generate subgoals using simple rule-based logic.
        """
        return self.fallback_plan(state)
