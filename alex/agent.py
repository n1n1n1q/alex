from __future__ import annotations

from typing import Dict, Any, Optional

from .core.types import GameState, SkillResult
from .core.extractor import extract_state
from .planning.metaplanner import MetaPlanner
from .planning.reflex import ReflexPolicy
from .planning.skill_router import SkillRouter
from .execution.policy_executor import execute_policy_skill
from .core.config import get_config

_config = get_config()

if _config.use_gemini_planner:
    try:
        from .planning.planner_gemini import GeminiPlanner as Planner
        print("✓ Using Gemini-based planner")
    except ImportError as e:
        print(f"⚠ Failed to import GeminiPlanner: {e}")
        print("  Falling back to simple rule-based planner")
        from .planning.planner import Planner
else:
    from .planning.planner import Planner
    if _config.verbose:
        print("ℹ Using simple rule-based planner (set GEMINI_API_KEY to use Gemini)")


class Agent:
    """Top-level orchestrator following the diagram:

    Game env -> state extraction -> reflex -> planner -> metaplanner -> skill response -> execution
    
    The agent can be configured through the AlexConfig system or by passing
    custom components for dependency injection.
    """

    def __init__(
        self,
        reflex: Optional[ReflexPolicy] = None,
        planner: Optional[Planner] = None,
        metaplanner: Optional[MetaPlanner] = None,
        router: Optional[SkillRouter] = None,
    ) -> None:
        """
        Initialize agent with optional dependency injection.
        
        Args:
            reflex: Custom reflex policy (default: ReflexPolicy())
            planner: Custom planner (default: based on config)
            metaplanner: Custom metaplanner (default: MetaPlanner())
            router: Custom skill router (default: SkillRouter())
        """
        self.reflex = reflex or ReflexPolicy()
        self.planner = planner or Planner()
        self.metaplanner = metaplanner or MetaPlanner()
        self.router = router or SkillRouter()

    def step(self, raw_obs: Dict[str, Any]) -> SkillResult:
        state: GameState = extract_state(raw_obs)

        reflex_goal = self.reflex.detect(state)
        if reflex_goal is not None:
            skill_req = self.router.to_skill(reflex_goal)
            return SkillResult(**execute_policy_skill(skill_req, env_obs=raw_obs))

        subgoals = self.planner.plan(state)
        backlog = self.metaplanner.update(subgoals)
        next_goal = self.metaplanner.pop_next()
        if next_goal is None:
            return SkillResult(status="OK", info={"note": "nothing to do"})

        skill_req = self.router.to_skill(next_goal)
        return SkillResult(**execute_policy_skill(skill_req, env_obs=raw_obs))
