from __future__ import annotations

from ..core.types import GameState, Subgoal


class ReflexPolicy:
    """Fast, local decision maker for urgent reactions.

    Placeholder rules only; replace with a small model if needed.
    """

    def detect(self, state: GameState) -> Subgoal | None:
        # If health is low, trigger emergency retreat or eat
        if state.health is not None and state.health <= 4:
            return Subgoal(name="emergency_retreat", params={}, priority=100)
        # If night and no shelter detected, suggest shelter
        # Note: time_of_day is not currently in GameState, so this check is disabled
        # Use inventory_agg instead of inventory for simple item counts
        # if state.time_of_day == "night" and state.inventory_agg.get("torch", 0) == 0:
        #     return Subgoal(name="seek_shelter", params={}, priority=90)
        return None
