from __future__ import annotations

from typing import Dict

from ..core.types import Subgoal, SkillRequest


class SkillRouter:
    """Map planner subgoals to concrete micro-skills (policy names).
    """

    def to_skill(self, subgoal: Subgoal) -> SkillRequest:
        mapping: Dict[str, str] = {
            "collect_wood": "gather_wood",
            "craft_table": "craft_table",
            "idle_scan": "idle_scan",
            "emergency_retreat": "retreat",
            "seek_shelter": "build_shelter",
            "block_in": "build_shelter",
            "eat_food": "gather_food",
            "fight_mob": "fight_mob",
        }
        skill_name = mapping.get(subgoal.name, subgoal.name)
        return SkillRequest(name=skill_name, params=subgoal.params)
