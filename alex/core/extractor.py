from __future__ import annotations

from typing import Any, Dict
import numpy as np
from PIL import Image

from .types import GameState
from ..utils.serialization import to_serializable, to_json_str as _to_json_str, to_json_file as _to_json_file

def _aggregate_inventory(inv: dict) -> dict:
    """Aggregate inventory counts by item type."""
    aggregated = {}
    for _, item in inv.items():
        item_type = item.get("type")
        quantity = item.get("quantity", 0) or 0
        if item_type and item_type not in ("none", "air") and quantity > 0:
            aggregated[item_type] = aggregated.get(item_type, 0) + quantity
    return aggregated


def extract_state(raw_info: Dict) -> GameState:
    """Convert environment observation dict to GameState."""
    info = raw_info or {}

    if not isinstance(info, dict) or not info:
        return GameState()

    env_state = {}

    env_state["biome_id"] = info.get("location_stats", {}).get("biome_id")
    env_state["biome_rainfall"] = info.get("location_stats", {}).get("biome_rainfall")
    env_state["biome_temperature"] = info.get("location_stats", {}).get("biome_temperature")
    env_state["can_see_sky"] = info.get("location_stats", {}).get("can_see_sky")
    env_state["is_raining"] = info.get("location_stats", {}).get("is_raining")
    env_state["light_level"] = info.get("location_stats", {}).get("light_level")
    env_state["sea_level"] = info.get("location_stats", {}).get("sea_level")
    env_state["sky_light_level"] = info.get("location_stats", {}).get("sky_light_level")
    env_state["sun_brightness"] = info.get("location_stats", {}).get("sun_brightness")

    player_pos = info.get("player_pos", {})

    blocks = info.get("voxels", [])

    mobs = info.get("mobs", [])

    health = info.get("health", 0)

    hunger = info.get("food_level", 0)

    is_gui_open = info.get("is_gui_open", False)

    inventory = info.get("inventory", {})

    inventory_agg = _aggregate_inventory(inventory)

    equipped_items = info.get("equipped_items", {})

    extras = {}

    extras["use_item"] = info.get("use_item", {})
    extras["pickup"] = info.get("pickup", {})
    extras["break_item"] = info.get("break_item", {})
    extras["craft_item"] = info.get("craft_item", {})
    extras["mine_block"] = info.get("mine_block", {})
    extras["damage_dealt"] = info.get("damage_dealt", {})
    extras["custom"] = info.get("custom", {})
    extras["kill_entity"] = info.get("kill_entity", {})

    return GameState(
        env_state=env_state,
        player_pos=player_pos,
        blocks=blocks,
        mobs=mobs,
        health=health,
        hunger=hunger,
        is_gui_open=is_gui_open,
        inventory=inventory,
        inventory_agg=inventory_agg,
        equipped_items=equipped_items,
        extras=extras,
    )

def extract_pov(raw_obs: Dict, raw_info: Dict, resized: bool = True) -> np.ndarray:
    """Extract POV image (H, W, 3) from raw observation or info."""
    image = None
    if resized:
        image = raw_obs.get("image")
    else:
        image = raw_info.get("pov")
    return image

def pov_to_image(pov: np.ndarray) -> Any:
    """Convert POV array to image format"""
    if pov is None:
        return None
    return Image.fromarray(pov)

def pov_to_image_file(pov: np.ndarray, filepath: str) -> None:
    """Save POV array as an image file."""
    if pov is None:
        return
    image = Image.fromarray(pov)
    image.save(filepath)

def state_to_json_str(state: GameState) -> str:
    """Convert GameState to JSON string."""
    return _to_json_str(state)

def state_to_json_file(state: GameState, filepath: str) -> None:
    """Save GameState to a JSON file."""
    _to_json_file(state, filepath)