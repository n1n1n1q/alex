from __future__ import annotations

from typing import Dict, Any
import os

from .types import GameState, SkillResult
from .extractor import extract_state
from .metaplanner import MetaPlanner
from .reflex import ReflexPolicy
from .skill_router import SkillRouter
from .policy_executor import execute_policy_skill

# Use Gemini planner if API key is available, otherwise fall back to simple planner
_USE_GEMINI = os.getenv("GEMINI_API_KEY") is not None

if _USE_GEMINI:
    try:
        from .planner_gemini import GeminiPlanner as Planner
        print("✓ Using Gemini-based planner")
    except ImportError as e:
        print(f"⚠ Failed to import GeminiPlanner: {e}")
        print("  Falling back to simple rule-based planner")
        from .planner import Planner
else:
    from .planner import Planner
    print("ℹ Using simple rule-based planner (set GEMINI_API_KEY to use Gemini)")


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
            # Pass raw_obs to enable STEVE-1 execution
            return SkillResult(**execute_policy_skill(skill_req, env_obs=raw_obs))

        # Planner + metaplanner
        subgoals = self.planner.plan(state)
        backlog = self.metaplanner.update(subgoals)
        next_goal = self.metaplanner.pop_next()
        if next_goal is None:
            return SkillResult(status="OK", info={"note": "nothing to do"})

        skill_req = self.router.to_skill(next_goal)
        # Pass raw_obs to enable STEVE-1 execution
        return SkillResult(**execute_policy_skill(skill_req, env_obs=raw_obs))
