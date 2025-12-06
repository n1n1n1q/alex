"""
Vision module for the Alex agent.

This package provides vision analysis capabilities for Minecraft using MineCLIP.
It includes:
- Image and text encoding (encoders.py)
- Scene analysis and categorization (scene_analyzer.py)
- Spatial attention and object localization (spatial_attention.py)
- Result formatting utilities (formatters.py)
"""

from .encoders import MineCLIPEncoder
from .scene_analyzer import SceneAnalyzer
from .spatial_attention import SpatialAttentionMap
from .formatters import VisionFormatter

# Import AgentVision from parent module
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from alex.agent_vision import AgentVision as _AgentVision
    AgentVision = _AgentVision
except ImportError:
    # Fallback if running from different context
    AgentVision = None

__all__ = [
    "MineCLIPEncoder",
    "SceneAnalyzer", 
    "SpatialAttentionMap",
    "VisionFormatter",
    "AgentVision",
]
