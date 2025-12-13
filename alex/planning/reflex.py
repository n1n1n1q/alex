from __future__ import annotations

from ..core.types import GameState, Subgoal


class ReflexPolicy:
    """
    Fast, local decision maker for urgent reactions.
    """

    def detect(self, state: GameState) -> Subgoal | None:
        if state.health is not None and state.health <= 4:
            return Subgoal(name="emergency_retreat", params={}, priority=100)
        return None
