from __future__ import annotations

from typing import List, Protocol
from abc import ABC, abstractmethod

from ..core.types import GameState, Subgoal


class BasePlanner(ABC):

    @abstractmethod
    def plan(self, state: GameState) -> List[Subgoal]:

        pass

    def fallback_plan(self, state: GameState) -> List[Subgoal]:

        subgoals: List[Subgoal] = []

        if state.health is not None and state.health <= 4:
            return [Subgoal(name="emergency_retreat", params={}, priority=100)]

        inv = state.inventory_agg if state.inventory_agg else {}

        if inv.get("planks", 0) < 4 and inv.get("log", 0) < 2:
            subgoals.append(
                Subgoal(name="collect_wood", params={"count": 4}, priority=10)
            )
        elif inv.get("crafting_table", 0) < 1:
            subgoals.append(Subgoal(name="craft_table", params={}, priority=8))
        else:
            subgoals.append(Subgoal(name="idle_scan", params={}, priority=0))

        return subgoals


__all__ = ["BasePlanner"]
