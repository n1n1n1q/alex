"""
Lightweight placeholder for RL skill execution.

The RL team will later replace the internals to call trained policies.
For now, this exposes a stable interface that the LLM side can depend on.

Contract
- Input: a SkillRequest-like payload (dict) with fields:
	- name: str  # canonical skill id, e.g. "gather_wood", "fight_mob"
	- params: dict  # any parameterization for the skill
	- timeout_ms: Optional[int]  # optional budget for synchronous execution
- Output: dict SkillResult with fields:
	- status: "OK" | "PENDING_POLICY" | "FAILED"
	- info: dict  # diagnostic details
	- low_level_actions: Optional[list]  # optional low-level env actions for debugging/scripting

This module intentionally avoids importing MineStudio directly so it can be
used in lightweight unit tests without the simulator. When the RL models are
ready, swap the body of `execute_policy_skill` to call into the trained policy
runner and return a structured SkillResult.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import time


@dataclass
class SkillRequest:
	name: str
	params: Dict[str, Any]
	timeout_ms: Optional[int] = None


def execute_policy_skill(request: SkillRequest | Dict[str, Any]) -> Dict[str, Any]:
	"""
	Execute a micro-skill via a (future) RL policy. Placeholder implementation.

	Parameters
	- request: SkillRequest or dict with 'name', 'params', optional 'timeout_ms'

	Returns
	- dict SkillResult as described in the module docstring.
	"""
	if not isinstance(request, SkillRequest):
		# Be forgiving to simple dict-based callers
		request = SkillRequest(
			name=str(request.get("name", "unknown")),
			params=dict(request.get("params", {})),
			timeout_ms=request.get("timeout_ms"),
		)

	# Placeholder: simulate work and return a deterministic no-op like action.
	start = time.time()
	# A tiny sleep to emulate “policy thinking” without blocking too long.
	time.sleep(min((request.timeout_ms or 10) / 1000.0, 0.01))

	# A generic no-op action that many env wrappers tolerate for stepping.
	# RL team: replace with actual action sequence produced by the policy.
	low_level_actions = [
		{"camera": [0.0, 0.0], "attack": 0, "back": 0, "forward": 0, "jump": 0,
		 "left": 0, "right": 0, "sneak": 0, "sprint": 0, "use": 0, "place": 0}
	]

	elapsed_ms = int((time.time() - start) * 1000)
	return {
		"status": "PENDING_POLICY",  # indicates the RL executor is a stub for now
		"info": {
			"skill": request.name,
			"params": request.params,
			"elapsed_ms": elapsed_ms,
			"note": "RL skill executor is a placeholder; replace with trained policy call.",
		},
		"low_level_actions": low_level_actions,
	}


__all__ = ["execute_policy_skill", "SkillRequest"]

