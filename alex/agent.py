from __future__ import annotations

from typing import Dict, Any

from .types import GameState, SkillResult
from .extractor import extract_state
from .planner import Planner
from .metaplanner import MetaPlanner
from .reflex import ReflexPolicy
from .skill_router import SkillRouter
from ..rl.policy_executor import execute_policy_skill


class Agent:
    """Top-level orchestrator following the diagram:

    Game env -> state extraction -> reflex -> planner -> metaplanner -> skill response -> execution
    """

    def __init__(self) -> None:
        self.reflex = ReflexPolicy()
        self.planner = Planner()
        self.metaplanner = MetaPlanner()
        self.router = SkillRouter()

    def step(self, raw_obs: Dict[str, Any]) -> SkillResult:
        state: GameState = extract_state(raw_obs)

        # Reflex stage: may preempt the planner
        reflex_goal = self.reflex.detect(state)
        if reflex_goal is not None:
            skill_req = self.router.to_skill(reflex_goal)
            return SkillResult(**execute_policy_skill(skill_req))

        # Planner + metaplanner
        subgoals = self.planner.plan(state)
        backlog = self.metaplanner.update(subgoals)
        next_goal = self.metaplanner.pop_next()
        if next_goal is None:
            return SkillResult(status="OK", info={"note": "nothing to do"})

        skill_req = self.router.to_skill(next_goal)
        return SkillResult(**execute_policy_skill(skill_req))
