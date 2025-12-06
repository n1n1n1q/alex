"""
Policy Skill Executor with STEVE-1 Integration

Executes micro-skills using STEVE-1 pretrained policy.

Flow:
1. Receive SkillRequest (high-level goal like "gather_wood")
2. Generate short STEVE-1 prompt using LLM ("mine log")
3. Execute STEVE-1 policy to generate low-level actions
4. Return SkillResult with action sequence

Contract:
- Input: SkillRequest with name, params, optional timeout_ms
- Output: SkillResult with status, info, and low_level_actions

The LLM (Gemini) generates appropriate STEVE-1 prompts based on the skill,
then STEVE-1 executes the low-level controller actions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import time
import os

# Import SkillRequest from types module instead of redefining it
from .types import SkillRequest


# Check if STEVE-1 and LLM prompt generation are available
_USE_STEVE = os.getenv("USE_STEVE_EXECUTOR", "false").lower() in ("true", "1", "yes")

try:
	from .steve_executor import execute_steve_action, STEVE_AVAILABLE
	from .action_prompt_generator import generate_steve_prompt, GEMINI_AVAILABLE
	_STEVE_READY = STEVE_AVAILABLE and GEMINI_AVAILABLE and _USE_STEVE
except ImportError:
	_STEVE_READY = False
	STEVE_AVAILABLE = False
	GEMINI_AVAILABLE = False


def execute_policy_skill(
request: SkillRequest | Dict[str, Any],
env_obs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""
Execute a micro-skill via STEVE-1 policy (if enabled) or placeholder.

Parameters:
- request: SkillRequest or dict with 'name', 'params', optional 'timeout_ms'
- env_obs: Current environment observation (required for STEVE-1 execution)

Returns:
- dict SkillResult with status, info, and low_level_actions

Environment Variable Configuration:
- USE_STEVE_EXECUTOR: Set to "true" to enable STEVE-1 execution
- GEMINI_API_KEY: Required for LLM prompt generation
"""
	if not isinstance(request, SkillRequest):
		# Be forgiving to simple dict-based callers
		request = SkillRequest(
name=str(request.get("name", "unknown")),
params=dict(request.get("params", {})),
timeout_ms=request.get("timeout_ms"),
)

	start = time.time()

	# STEVE-1 execution path
	if _STEVE_READY and env_obs is not None:
		try:
			# Step 1: Generate STEVE-1 prompt using LLM
			steve_prompt = generate_steve_prompt(request.name, request.params)
			
			# Step 2: Calculate max_steps from timeout_ms
			max_steps = 100  # default
			if request.timeout_ms:
				# Rough estimate: 20ms per step in Minecraft
				max_steps = min(max(request.timeout_ms // 20, 10), 500)
			
			# Step 3: Execute STEVE-1 policy
			result = execute_steve_action(
text_command=steve_prompt,
env_obs=env_obs,
max_steps=max_steps,
cond_scale=5.0,  # Default conditioning scale
)
			
			# Add timing info
			elapsed_ms = int((time.time() - start) * 1000)
			result["info"]["elapsed_ms"] = elapsed_ms
			result["info"]["skill"] = request.name
			result["info"]["skill_params"] = request.params
			result["info"]["steve_prompt"] = steve_prompt
			result["info"]["execution_mode"] = "STEVE-1"
			
			return result
			
		except Exception as e:
			# Fall back to placeholder on error
			elapsed_ms = int((time.time() - start) * 1000)
			return {
				"status": "FAILED",
				"info": {
					"skill": request.name,
					"params": request.params,
					"elapsed_ms": elapsed_ms,
					"error": str(e),
					"steve_prompt": steve_prompt if 'steve_prompt' in locals() else None,
					"note": "STEVE-1 execution failed, see error details",
				},
				"low_level_actions": [],
			}
	
	# Placeholder path (original behavior)
	# A tiny sleep to emulate "policy thinking" without blocking too long.
	time.sleep(min((request.timeout_ms or 10) / 1000.0, 0.01))

	# A generic no-op action that many env wrappers tolerate for stepping.
	low_level_actions = [
		{"camera": [0.0, 0.0], "attack": 0, "back": 0, "forward": 0, "jump": 0,
		 "left": 0, "right": 0, "sneak": 0, "sprint": 0, "use": 0, "place": 0}
	]

	elapsed_ms = int((time.time() - start) * 1000)
	
	status_msg = "PENDING_POLICY"
	note = "RL skill executor is a placeholder"
	
	if not _USE_STEVE:
		note += " (USE_STEVE_EXECUTOR not enabled)"
	elif not STEVE_AVAILABLE:
		note += " (MineStudio not installed)"
	elif not GEMINI_AVAILABLE:
		note += " (google-generativeai not installed)"
	elif env_obs is None:
		note += " (no env_obs provided)"
	
	return {
		"status": status_msg,
		"info": {
			"skill": request.name,
			"params": request.params,
			"elapsed_ms": elapsed_ms,
			"note": note,
			"execution_mode": "placeholder",
		},
		"low_level_actions": low_level_actions,
	}


__all__ = ["execute_policy_skill", "SkillRequest"]
