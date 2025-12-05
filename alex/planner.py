from __future__ import annotations

from typing import List

from .types import GameState, Subgoal


class Planner:
    """Very simple rule-based planner placeholder.

    LLM team: replace the `plan` method with an actual LLM call that takes a
    JSON-serializable state and returns subgoals. Keep the Subgoal schema.
    """

    def plan(self, state: GameState) -> List[Subgoal]:
        subgoals: List[Subgoal] = []
        # Naive heuristics to get the data flow moving
        if (state.inventory.get("planks", 0) < 4) and (state.inventory.get("logs", 0) < 2):
            subgoals.append(Subgoal(name="collect_wood", params={"count": 4}, priority=10))
        elif state.inventory.get("crafting_table", 0) < 1:
            subgoals.append(Subgoal(name="craft_table", params={}, priority=8))
        else:
            subgoals.append(Subgoal(name="idle_scan", params={}, priority=0))
        return subgoals
