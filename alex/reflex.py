from __future__ import annotations

from .types import GameState, Subgoal


class ReflexPolicy:
    """Fast, local decision maker for urgent reactions.

    Placeholder rules only; replace with a small model if needed.
    """

    def detect(self, state: GameState) -> Subgoal | None:
        # If health is low, trigger emergency retreat or eat
        if state.health is not None and state.health <= 4:
            return Subgoal(name="emergency_retreat", params={}, priority=100)
        # If night and no shelter detected, suggest shelter
        if state.time_of_day == "night" and state.inventory.get("torch", 0) == 0:
            return Subgoal(name="seek_shelter", params={}, priority=90)
        return None
