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

if _config.use_hf_planner:
    try:
        from .planning.hf_planner import HuggingFacePlanner as Planner
        if _config.verbose:
            print("Using HuggingFace-based planner")
    except ImportError as e:
        if _config.verbose:
            print(f"Failed to import HuggingFacePlanner: {e}")
            print("  Falling back to simple rule-based planner")
        from .planning.planner import Planner
else:
    from .planning.planner import Planner
    if _config.verbose:
        print("Using simple rule-based planner (set USE_HF_PLANNER=true to use HuggingFace)")


class Agent:

    def __init__(
        self,
        reflex: Optional[ReflexPolicy] = None,
        planner: Optional[Planner] = None,
        metaplanner: Optional[MetaPlanner] = None,
        router: Optional[SkillRouter] = None,
    ) -> None:

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


class VerboseAgent(Agent):
    def step(self, raw_obs: Dict[str, Any]) -> SkillResult:
        state = extract_state(raw_obs)
        
        print(f"  [Reflex] Checking for urgent situations...")
        # reflex_goal = self.reflex.detect(state)
        # if reflex_goal is not None:
        #     print(f"  [Reflex] TRIGGERED: {reflex_goal}")
        #     skill_req = self.router.to_skill(reflex_goal)
        #     print(f"  [Router] Mapped to skill: {skill_req}")
        #     result = execute_policy_skill(skill_req, env_obs=raw_obs)
        #     print(f"  [Executor] Result: {result.get('status', 'unknown')}")
        #     return SkillResult(**result)
        
        print(f"  [Reflex] No urgent situations detected")
        
        print(f"  [Planner] Ensuring resources are loaded...")
        self.planner._ensure_resources()
        print(f"  [Planner] Analyzing state and generating subgoals...")
        subgoals = self.planner.plan(state)
        print(f"  [Planner] Generated {len(subgoals)} subgoal(s):")
        for sg in subgoals:
            print(f"    - {sg.name} (priority={sg.priority}, params={sg.params})")
        
        print(f"  [MetaPlanner] Updating backlog...")
        backlog = self.metaplanner.update(subgoals)
        print(f"  [MetaPlanner] Current backlog ({len(backlog)} items):")
        for i, sg in enumerate(backlog[:5], 1):
            print(f"    {i}. {sg.name} (priority={sg.priority})")
        if len(backlog) > 5:
            print(f"    ... and {len(backlog) - 5} more")
        
        next_goal = self.metaplanner.pop_next()
        if next_goal is None:
            print(f"  [MetaPlanner] No goals in backlog - nothing to do")
            return SkillResult(status="OK", info={"note": "nothing to do"})
        
        print(f"  [MetaPlanner] Selected next goal: {next_goal.name}")
        
        print(f"  [Router] Routing goal to skill...")
        skill_req = self.router.to_skill(next_goal)
        print(f"  [Router] Skill request: {skill_req}")
        
        print(f"  [Executor] Executing skill...")
        result = execute_policy_skill(skill_req, env_obs=raw_obs)
        print(f"  [Executor] Status: {result.get('status', 'unknown')}")
        if result.get('info', {}).get('steve_prompt'):
            print(f"  [Executor] STEVE-1 Prompt: '{result['info']['steve_prompt']}'")
        
        return SkillResult(**result)
