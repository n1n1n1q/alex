from __future__ import annotations

from typing import Any, Dict

from .types import GameState


def extract_state(raw_obs: Dict[str, Any]) -> GameState:
    """Convert environment observation dict to GameState.

    This is a minimal placeholder; wire this to MineStudio's observation keys
    later. Keep the output JSON-friendly.
    """
    inv = raw_obs.get("inventory", {}) if isinstance(raw_obs, dict) else {}
    return GameState(
        pov_image=raw_obs.get("pov"),
        inventory={k: int(v) for k, v in inv.items()} if isinstance(inv, dict) else {},
        health=raw_obs.get("health"),
        hunger=raw_obs.get("hunger"),
        position=raw_obs.get("position"),
        biome=raw_obs.get("biome"),
        time_of_day=raw_obs.get("time_of_day"),
        nearby_entities=raw_obs.get("nearby_entities", []),
        extras={k: v for k, v in raw_obs.items() if k not in {"pov","inventory","health","hunger","position","biome","time_of_day","nearby_entities"}},
    )
