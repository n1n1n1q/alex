from __future__ import annotations

from typing import List

from .types import Subgoal


class MetaPlanner:
    """Tracks long-horizon context and reorders/prunes subgoals.

    Placeholder logic keeps the highest priority first.
    """

    def __init__(self) -> None:
        self.backlog: List[Subgoal] = []

    def update(self, new_subgoals: List[Subgoal]) -> List[Subgoal]:
        self.backlog.extend(new_subgoals)
        # Simple prioritization
        self.backlog.sort(key=lambda s: s.priority, reverse=True)
        # Deduplicate by name (keep highest priority)
        unique = {}
        for s in self.backlog:
            if s.name not in unique:
                unique[s.name] = s
        self.backlog = list(unique.values())
        return list(self.backlog)

    def pop_next(self) -> Subgoal | None:
        if not self.backlog:
            return None
        return self.backlog.pop(0)
