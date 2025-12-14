from __future__ import annotations

from typing import Dict, Any, Optional
from unittest import result

from .core.types import GameState, SkillResult
from .core.extractor import extract_state
from .planning.metaplanner import MetaPlanner
from .planning.skill_router import SkillRouter
from .planning.reflex import ReflexPolicy
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
        print(
            "Using simple rule-based planner (enable use_hf_planner in config to use HuggingFace)"
        )

if _config.use_hf_reflex_manager:
    try:
        from .planning.hf_reflex_manager import (
            HuggingFaceReflexManager as ReflexManager,
        )

        if _config.verbose:
            print("Using HuggingFace-based reflex manager")
    except ImportError as e:
        if _config.verbose:
            print(f"Failed to import HuggingFaceReflexManager: {e}")
            print("  Falling back to simple rule-based reflex")
        from .planning.reflex import ReflexPolicy as ReflexManager
else:
    from .planning.reflex import ReflexPolicy as ReflexManager

    if _config.verbose:
        print(
            "Using simple rule-based reflex (enable use_hf_reflex_manager in config to use HuggingFace)"
        )


class Agent:

    def __init__(
        self,
        reflex: Optional[ReflexPolicy] = None,
        planner: Optional[Planner] = None,
        metaplanner: Optional[MetaPlanner] = None,
        router: Optional[SkillRouter] = None,
    ) -> None:

        self.reflex = reflex or ReflexManager()
        self.planner = planner or Planner()
        self.metaplanner = metaplanner or MetaPlanner()
        self.router = router or SkillRouter()


        self.current_goal = None

    def step(
        self, raw_obs: Dict[str, Any], state: Optional[GameState] = None
    ) -> SkillResult:
        if state is None:
            state: GameState = extract_state(raw_obs)

        reflex_goal = self.reflex.detect(state)
        if reflex_goal is not None:
            skill_req = self.router.to_skill(reflex_goal)
            result = execute_policy_skill(skill_req, env_obs=raw_obs)

            if (
                hasattr(self.reflex, "_last_raw_response")
                and self.reflex._last_raw_response
            ):
                if "info" not in result:
                    result["info"] = {}
                result["info"]["reflex_response"] = {
                    "raw": self.reflex._last_raw_response,
                    "parsed": getattr(self.reflex, "_last_parsed_response", None),
                }

            return SkillResult(**result)

        subgoals = self.planner.plan(state)
        backlog = self.metaplanner.update(subgoals)
        next_goal = self.metaplanner.pop_next()
        if next_goal is None:
            return SkillResult(status="OK", info={"note": "nothing to do"})

        skill_req = self.router.to_skill(next_goal)
        result = execute_policy_skill(skill_req, env_obs=raw_obs)

        if (
            hasattr(self.planner, "_last_raw_response")
            and self.planner._last_raw_response
        ):
            if "info" not in result:
                result["info"] = {}
            result["info"]["raw_model_response"] = self.planner._last_raw_response
            result["info"]["parsed_plan"] = {
                "cleaned_json": getattr(self.planner, "_last_cleaned_response", None),
                "subgoals": [
                    {"name": sg.name, "priority": sg.priority, "params": sg.params}
                    for sg in subgoals
                ],
            }

        return SkillResult(**result)


class VerboseAgent(Agent):
    def step(self, raw_obs: Dict[str, Any], state: GameState) -> SkillResult:
        print(f"  [Reflex] Checking for urgent situations...")
        print(f"  [Reflex] No urgent situations detected")

        self.planner._ensure_resources()
        print(f"  [Planner] Analyzing state and generating subgoals...")

        state.extras = {'current_goal': self.current_goal}

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

        result = {'status': 'SUCCESS'}

        result['info'] = {'steve_prompt': next_goal.name} 

        self.current_goal = next_goal.name

        if (
            hasattr(self.planner, "_last_raw_response")
            and self.planner._last_raw_response
        ):
            if "info" not in result:
                result["info"] = {}
            result["info"]["raw_model_response"] = self.planner._last_raw_response
            result["info"]["parsed_plan"] = {
                "cleaned_json": getattr(self.planner, "_last_cleaned_response", None),
                "subgoals": [
                    {"name": sg.name, "priority": sg.priority, "params": sg.params}
                    for sg in subgoals
                ],
            }

        return SkillResult(**result)
