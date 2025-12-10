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

__all__ = [
    "MineCLIPEncoder",
    "SceneAnalyzer", 
    "SpatialAttentionMap",
    "VisionFormatter",
]
