from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GameState:
    """Canonical, LLM-friendly state snapshot.

    Keep it light and JSON-serializable so it can be sent to an LLM or MCP.
    """
    pov_image: Optional[Any] = None  # placeholder: RGB array or bytes
    inventory: Dict[str, int] = field(default_factory=dict)
    health: Optional[float] = None
    hunger: Optional[float] = None
    position: Optional[Dict[str, float]] = None  # {x, y, z}
    biome: Optional[str] = None
    time_of_day: Optional[str] = None  # day/night/dawn/dusk
    nearby_entities: List[Dict[str, Any]] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Subgoal:
    """A structured subgoal produced by the Planner.

    name: canonical id (e.g., "collect_wood", "craft_table")
    params: goal-specific args (e.g., count=8)
    priority: higher means more urgent
    """
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0


@dataclass
class SkillRequest:
    """Request to execute a micro-skill.

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
