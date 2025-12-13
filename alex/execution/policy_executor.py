from __future__ import annotations

import torch

import numpy as np
from typing import Any, Dict, Optional
import time

from ..core.types import SkillRequest
from ..core.config import get_config

_config = get_config()
_STEVE_READY = False

if _config.use_steve_executor:
    try:
        from .steve_executor import SteveExecutor, STEVE_AVAILABLE
        from ..prompts.action_prompt_generator import (
            ActionPromptGenerator,
            HF_AVAILABLE,
        )

        _STEVE_READY = STEVE_AVAILABLE and HF_AVAILABLE
    except ImportError:
        STEVE_AVAILABLE = False
        HF_AVAILABLE = False


_steve_executor: Optional[Any] = None
_prompt_generator: Optional[Any] = None


def _get_steve_executor():

    global _steve_executor
    if _steve_executor is None and _STEVE_READY:
        _steve_executor = SteveExecutor(
            model_path=_config.steve_model_path,
            default_cond_scale=_config.steve_default_cond_scale,
            default_max_steps=_config.steve_default_max_steps,
        )
    return _steve_executor


def _get_prompt_generator():

    global _prompt_generator
    if _prompt_generator is None and _STEVE_READY:
        _prompt_generator = ActionPromptGenerator(
            model_name=_config.hf_model_name,
            device=_config.device,
            verbose=_config.verbose,
        )
    return _prompt_generator


def execute_policy_skill(
    request: SkillRequest | Dict[str, Any],
    env_obs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    if not isinstance(request, SkillRequest):
        request = SkillRequest(
            name=str(request.get("name", "unknown")),
            params=dict(request.get("params", {})),
            timeout_ms=request.get("timeout_ms"),
        )

    start = time.time()

    if _STEVE_READY and env_obs is not None:
        try:
            executor = _get_steve_executor()

            if executor is None:
                raise RuntimeError("Failed to initialize STEVE components")

            steve_prompt = (
                request.name
            )  # generator.generate_prompt(request.name, request.params)

            max_steps = 100
            if request.timeout_ms:
                max_steps = min(max(request.timeout_ms // 20, 10), 500)

            env_obs["image"] = torch.from_numpy(env_obs["image"]).permute(2, 0, 1)

            result = executor.execute(
                text_command=request.name,
                env_obs=env_obs,
                max_steps=max_steps,
                cond_scale=_config.steve_default_cond_scale,
            )

            elapsed_ms = int((time.time() - start) * 1000)
            result["info"]["elapsed_ms"] = elapsed_ms
            result["info"]["skill"] = request.name
            result["info"]["skill_params"] = request.params
            result["info"]["steve_prompt"] = steve_prompt
            result["info"]["execution_mode"] = "STEVE-1"
            print(">>> Immediate result", result)
            return result

        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)

            import traceback

            traceback.print_exc()

            return {
                "status": "FAILED",
                "info": {
                    "skill": request.name,
                    "params": request.params,
                    "elapsed_ms": elapsed_ms,
                    "error": str(e),
                    "steve_prompt": (
                        steve_prompt if "steve_prompt" in locals() else None
                    ),
                    "note": "STEVE-1 execution failed, see error details",
                },
                "low_level_actions": [],
            }

    time.sleep(min((request.timeout_ms or 10) / 1000.0, 0.01))

    low_level_actions = [
        {
            "camera": [0.0, 0.0],
            "attack": 0,
            "back": 0,
            "forward": 0,
            "jump": 0,
            "left": 0,
            "right": 0,
            "sneak": 0,
            "sprint": 0,
            "use": 0,
            "place": 0,
        }
    ]

    elapsed_ms = int((time.time() - start) * 1000)

    status_msg = "PENDING_POLICY"
    note = "RL skill executor is a placeholder"

    if not _config.use_steve_executor:
        note += " (use_steve_executor not enabled in config)"
    elif not _STEVE_READY:
        note += " (STEVE components not available)"
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
