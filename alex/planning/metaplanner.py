from __future__ import annotations

from typing import List

from ..core.types import Subgoal


class MetaPlanner:
    """
    Tracks long-horizon context and reorders/prunes subgoals.
    Placeholder logic keeps the highest priority first.
    """

    def __init__(self) -> None:
        self.backlog: List[Subgoal] = []

    def update(self, new_subgoals: List[Subgoal]) -> List[Subgoal]:

        for new_subgoal in new_subgoals:
            if new_subgoal not in self.backlog:
                self.backlog.append(new_subgoal) 

        return self.backlog

    def pop_next(self) -> Subgoal | None:
        if not self.backlog:
            return None
        return self.backlog.pop(0)
