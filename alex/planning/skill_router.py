from __future__ import annotations

from typing import Dict

from ..core.types import Subgoal, SkillRequest


class SkillRouter:
    """Map planner subgoals to concrete micro-skills (policy names).
    * Currently a 1:1 mapping with identical names and params.
    """

    def to_skill(self, subgoal: Subgoal) -> SkillRequest:
        return SkillRequest(name=subgoal.name, params=subgoal.params)
