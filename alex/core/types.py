from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GameState:
    """
    State snapshot of the Minecraft environment.

    env_state: biome id plus rainfall/temperature, sky & block light, rain flag, sea level, whether the sky is visible, sun brightness
    player_pos: agent's coordinates plus yaw/pitch

    blocks: nearby blocks data (if requested)
    mobs: nearby entities data (if requested)

    health: current health points (0-20 in survival)
    hunger: current hunger bar state (0-20 in survival)

    is_gui_open: GUI currently open flag
    inventory: 36-slot map (0-8 hotbar, 9-35 main inventory) where each entry contains the item id and stack count
    inventory_agg: aggregated inventory counts by item type
    equipped_items: per-slot armour/offhand/mainhand payload with type, current damage, and maxDamage

    extras: catch-all for any other info keys
    """

    env_state: Dict[str, Any] = field(default_factory=dict)
    player_pos: Dict[str, Any] = field(default_factory=dict)

    blocks: List[Any] = field(default_factory=list)
    mobs: List[Dict[str, Any]] = field(default_factory=list)

    health: Optional[float] = None
    hunger: Optional[int] = None

    is_gui_open: Optional[bool] = None
    inventory: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    inventory_agg: Dict[str, int] = field(default_factory=dict)
    equipped_items: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Subgoal:
    """
    A structured subgoal produced by the Planner.

    name: canonical id (e.g., "collect_wood", "craft_table")
    params: goal-specific args (e.g., count=8)
    priority: higher means more urgent
    """

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0


@dataclass
class SkillRequest:
    """
    Request to execute a micro-skill.

    Typically derived from a Subgoal by the SkillRouter.
    """

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: Optional[int] = None


@dataclass
class SkillResult:
    status: str  # OK | FAILED | PENDING_POLICY
    info: Dict[str, Any] = field(default_factory=dict)
    low_level_actions: Optional[List[Dict[str, Any]]] = None
