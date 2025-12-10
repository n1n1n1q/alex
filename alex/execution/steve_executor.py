"""
STEVE-1 Policy Executor

Integrates STEVE-1 (pretrained Minecraft VLM policy) for low-level action execution.
STEVE-1 takes short text prompts and executes goal-oriented behavior.

Key Design:
- Receives text command (e.g., "mine log", "kill cow")
- Executes STEVE-1 policy for specified number of steps
- Returns low-level action sequence
- Supports configurable conditioning scale and timeouts

IMPORTANT: STEVE-1 Limitations
- Pure movement prompts ("move forward", "run") typically FAIL
- Only GOAL-ORIENTED prompts work well ("mine log", "kill cow", "craft table")
- Trained on YouTube videos with specific object-focused behavior
- Use 2-3 word commands with Minecraft terminology
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from minestudio.models import SteveOnePolicy as SteveOnePolicyType
else:
    SteveOnePolicyType = Any

import numpy as np

try:
    from minestudio.simulator.callbacks import MinecraftCallback
    from minestudio.models import SteveOnePolicy
    STEVE_AVAILABLE = True
except ImportError:
    STEVE_AVAILABLE = False
    MinecraftCallback = object  # type: ignore
    SteveOnePolicy = None  # type: ignore


class CommandCallback(MinecraftCallback):
    """
    Injects text command into observations for STEVE-1 conditioning.
    """
    
    def __init__(self, command: str, cond_scale: float = 4.0):
        """
        Args:
            command: Text prompt for STEVE-1 (e.g., "mine log", "kill pig")
            cond_scale: Conditioning strength (2.0-8.0). Higher = stricter following
        """
        super().__init__()
        self.command = command
        self.cond_scale = cond_scale
        self.timestep = 0

    def after_reset(self, sim, obs, info):
        """Add condition to observation after environment reset."""
        self.timestep = 0
        if 'condition' not in obs:
            obs['condition'] = {}
        obs['condition'] = {
            "cond_scale": self.cond_scale,
            "text": self.command
        }
        return obs, info
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        """Add condition to observation after each step."""
        self.timestep += 1
        if 'condition' not in obs:
            obs['condition'] = {}
        obs['condition'] = {
            "cond_scale": self.cond_scale,
            "text": self.command
        }
        return obs, reward, terminated, truncated, info


class SteveExecutor:
    """
    Executes STEVE-1 policy to generate low-level actions from text commands.
    """
    
    def __init__(
        self,
        model_path: str = "CraftJarvis/MineStudio_STEVE-1.official",
        default_cond_scale: float = 5.0,
        default_max_steps: int = 100,
    ):
        """
        Initialize STEVE-1 executor.
        
        Args:
            model_path: HuggingFace path to STEVE-1 model
            default_cond_scale: Default conditioning strength (2.0-8.0)
            default_max_steps: Default maximum steps per execution
        """
        if not STEVE_AVAILABLE:
            raise ImportError(
                "MineStudio not available. Install with: "
                "pip install git+https://github.com/annastasyshyn/MineStudio.git"
            )
        
        self.model_path = model_path
        self.default_cond_scale = default_cond_scale
        self.default_max_steps = default_max_steps
        self.policy = None  # SteveOnePolicy instance (loaded lazily)
        
    def load_policy(self) -> None:
        """Lazy load STEVE-1 policy (heavy operation, call once)."""
        if self.policy is None:
            self.policy = SteveOnePolicy.from_pretrained(self.model_path)
    
    def execute(
        self,
        text_command: str,
        env_obs: Dict[str, Any],
        max_steps: Optional[int] = None,
        cond_scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute STEVE-1 policy for given text command.
        
        Args:
            text_command: Short goal-oriented prompt (e.g., "mine log", "kill cow")
            env_obs: Current environment observation from MinecraftSim
            max_steps: Maximum steps to execute (defaults to default_max_steps)
            cond_scale: Conditioning strength (defaults to default_cond_scale)
            
        Returns:
            Dict with:
                - status: "OK" | "FAILED"
                - low_level_actions: List of action dicts
                - info: Execution metadata
        """
        self.load_policy()
        
        max_steps = max_steps or self.default_max_steps
        cond_scale = cond_scale or self.default_cond_scale
        
        # Add condition to observation
        obs = env_obs.copy()
        if 'condition' not in obs:
            obs['condition'] = {}
        obs['condition'] = {
            "cond_scale": cond_scale,
            "text": text_command
        }
        
        actions = []
        state_in = None
        
        try:
            for step in range(max_steps):
                action, state_in = self.policy.get_action(obs, state_in)
                actions.append(action)
                
            return {
                "status": "OK",
                "low_level_actions": actions,
                "info": {
                    "text_command": text_command,
                    "cond_scale": cond_scale,
                    "steps_executed": len(actions),
                    "max_steps": max_steps,
                }
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "low_level_actions": actions,
                "info": {
                    "error": str(e),
                    "text_command": text_command,
                    "steps_executed": len(actions),
                }
            }
    
    def unload_policy(self) -> None:
        """Unload policy to free GPU memory."""
        self.policy = None


__all__ = [
    "SteveExecutor",
    "CommandCallback",
    "STEVE_AVAILABLE",
]
