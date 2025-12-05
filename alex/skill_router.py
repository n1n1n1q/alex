from __future__ import annotations

from typing import Dict

from .types import Subgoal, SkillRequest


class SkillRouter:
    """Map planner subgoals to concrete micro-skills (policy names).

    This is where naming conventions and parameter shaping live.
    """

    def to_skill(self, subgoal: Subgoal) -> SkillRequest:
        mapping: Dict[str, str] = {
            "collect_wood": "gather_wood",
            "craft_table": "craft_table",
            "idle_scan": "idle_scan",
            "emergency_retreat": "retreat",
            "seek_shelter": "build_shelter",
        }
        skill_name = mapping.get(subgoal.name, subgoal.name)
        return SkillRequest(name=skill_name, params=subgoal.params)
