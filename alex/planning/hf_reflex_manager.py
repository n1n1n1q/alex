from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import torch

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    HF_AVAILABLE = False

from ..core.config import get_config
from ..core.types import GameState, Subgoal
from .reflex import ReflexPolicy


class HuggingFaceReflexManager:
    """
    Low-latency reflex layer backed by a small transformers model.

    Responsibilities:
    - Make an immediate, local decision when obvious danger/opportunity is present.
    - Emit at most one Subgoal; fall back to rule-based reflexes on failure.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        if not HF_AVAILABLE:
            raise ImportError("transformers not installed. Install with `pip install transformers torch`.")

        cfg = get_config()
        self.verbose = verbose
        self.model_name = model_name or cfg.hf_reflex_model_name or cfg.hf_model_name
        self.device = device or cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature if temperature is not None else cfg.hf_reflex_temperature
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else cfg.hf_reflex_max_tokens

        if self.verbose:
            print(f"[HuggingFaceReflexManager] Loading model: {self.model_name} on {self.device}...")

        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
        )

        self.generation_config: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False,
        }

        self.fallback_reflex = ReflexPolicy()

        self.system_prompt = (
            "You are a fast, risk-averse reflex manager for a Minecraft agent. "
            "When danger or urgent need is detected, pick ONE action from the allowed list. "
            "Respond ONLY with JSON. Keep decisions localized and conservative."
        )
        self.allowed_actions = {
            "emergency_retreat": {"priority": 100, "description": "Sprint away from imminent lethal risk."},
            "block_in": {"priority": 90, "description": "Place blocks to create cover / hole for safety."},
            "eat_food": {"priority": 70, "description": "Consume or seek food if hunger critically low."},
            "fight_mob": {"priority": 80, "description": "Engage threatening hostile mob in close range."},
            "scan_area": {"priority": 40, "description": "Pause briefly and scan surroundings."},
            "none": {"priority": 0, "description": "No immediate reflex needed."},
        }

    def detect(self, state: GameState) -> Optional[Subgoal]:
        """
        Returns a Subgoal if an urgent reflex is warranted, else None.
        """
        try:
            state_summary = self._summarize_state(state)
            user_prompt = self._build_user_prompt(state_summary)

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            if self.verbose:
                print("\n[ReflexManager] Evaluating state for reflex...")

            outputs = self.pipe(messages, **self.generation_config)
            raw_text = outputs[0]["generated_text"]
            cleaned = self._extract_json(raw_text)
            parsed = json.loads(cleaned)

            action = str(parsed.get("action", "none")).strip().lower()
            params = parsed.get("params", {}) or {}
            priority = int(parsed.get("priority", self.allowed_actions.get(action, {}).get("priority", 50)))

            if self.verbose:
                print(f"[ReflexManager] Model action: {action}, priority={priority}, params={params}")

            subgoal = self._action_to_subgoal(action, params, priority)
            return subgoal
        except Exception as exc:  # pragma: no cover - defensive catch
            if self.verbose:
                print(f"[ReflexManager] LLM reflex failed ({exc}); falling back to rule-based reflex.")
            return self.fallback_reflex.detect(state)

    def _summarize_state(self, state: GameState) -> Dict[str, Any]:
        mobs = state.mobs or []
        nearest_mob = mobs[0] if mobs else None

        summary = {
            "health": state.health,
            "hunger": state.hunger,
            "has_shield": any(slot.get("item_id") == "minecraft:shield" for slot in state.equipped_items.values()),
            "inventory": state.inventory_agg,
            "nearest_hostile": nearest_mob["name"] if nearest_mob and nearest_mob.get("is_hostile") else None,
            "nearest_hostile_dist": nearest_mob.get("distance") if nearest_mob and nearest_mob.get("is_hostile") else None,
            "is_raining": state.env_state.get("is_raining") if state.env_state else None,
        }
        return {k: v for k, v in summary.items() if v is not None}

    def _build_user_prompt(self, state_summary: Dict[str, Any]) -> str:
        allowed = "\n".join([f"- {k}: {v['description']}" for k, v in self.allowed_actions.items()])
        return (
            "Decide if an immediate reflex action is required. "
            "If none is needed, choose 'none'.\n\n"
            f"Allowed actions:\n{allowed}\n\n"
            f"Current state (JSON): {json.dumps(state_summary)}\n\n"
            'Respond with JSON: {"action": "<one action>", "priority": <int>, "params": {...}, "reason": "<short>"}'
        )

    def _extract_json(self, text: str) -> str:
        text = text.strip()
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def _action_to_subgoal(self, action: str, params: Dict[str, Any], priority: int) -> Optional[Subgoal]:
        if action in ("none", "hold", "wait", "noop"):
            return None

        mapping = {
            "emergency_retreat": Subgoal(name="emergency_retreat", params=params, priority=priority),
            "block_in": Subgoal(name="seek_shelter", params=params, priority=priority),
            "eat_food": Subgoal(name="eat_food", params=params, priority=priority),
            "fight_mob": Subgoal(
                name="fight_mob",
                params={"mob": params.get("mob") or "hostile"},
                priority=priority,
            ),
            "scan_area": Subgoal(name="idle_scan", params=params, priority=priority),
        }

        return mapping.get(action)


__all__ = ["HuggingFaceReflexManager", "HF_AVAILABLE"]

