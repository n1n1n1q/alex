from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import numpy as np
from PIL import Image

# Add MineCLIP to path
MINECLIP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../submodules/MineCLIP'))
if MINECLIP_PATH not in sys.path:
    sys.path.insert(0, MINECLIP_PATH)

from mineclip import MineCLIP

from .vision.encoders import MineCLIPEncoder
from .vision.scene_analyzer import SceneAnalyzer
from .vision.spatial_attention import SpatialAttentionMap
from .vision.formatters import VisionFormatter


class AgentVision:
    """
    Vision module for the Minecraft agent using MineCLIP.
    
    Provides direct access to vision components:
    - encoder: MineCLIPEncoder for image/text encoding
    - scene_analyzer: SceneAnalyzer for comprehensive scene analysis
    - spatial_attention: SpatialAttentionMap for spatial object detection
    - formatter: VisionFormatter for output formatting
    
    For most use cases, use analyze_comprehensive() which combines
    both global scene analysis and spatial detection.
    """
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the vision module with MineCLIP.
        
        Args:
            weights_path: Path to MineCLIP weights. If None, defaults to ../models/avg.pth
            device: Device to run on ('cpu', 'cuda', 'mps'). Auto-detects if None.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        if weights_path is None:
            weights_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                '../models/avg.pth'
            ))
        else:
            weights_path = os.path.abspath(weights_path)
        
        if "avg" in os.path.basename(weights_path).lower():
            pool_type = "avg"
        else:
            pool_type = "attn.d2.nh8.glusw"
        
        self.model = MineCLIP(
            arch="vit_base_p16_fz.v2.t2",
            resolution=(160, 256),
            pool_type=pool_type,
            image_feature_dim=512,
            mlp_adapter_spec="v0-2.t0",
            hidden_dim=512,
        ).to(device)
        
        if os.path.exists(weights_path):
            self.model.load_ckpt(weights_path, strict=True)
            print(f"✓ Loaded MineCLIP weights: {weights_path}")
        else:
            print(f"⚠ Warning: Weights not found at {weights_path}")
            print(f"  Model initialized but not loaded. Vision may not work properly.")
        
        self.model.eval()
        
        self.encoder = MineCLIPEncoder(self.model, device)
        self.scene_analyzer = SceneAnalyzer(self.encoder)
        self.spatial_attention = SpatialAttentionMap(self.model, device)
        self.formatter = VisionFormatter()
    
    def analyze_comprehensive(self, image: Image.Image) -> dict:
        """
        Perform both global scene analysis AND spatial analysis.
        
        This is the recommended high-level method for most use cases.
        It combines scene categorization with spatial object detection.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary with both global and spatial analysis:
            - global: Scene categorization (biome, time, mobs, resources, etc.)
            - spatial: Spatial object detection with locations
            - combined_context: Human-readable text combining both analyses
        """
        global_analysis = self.scene_analyzer.analyze_comprehensive(image)
        
        from .vision.vision_queries import SPATIAL_QUERIES
        image_tensor = self.encoder.preprocess_image_to_tensor(image)
        spatial_analysis = self.spatial_attention.analyze_spatial(
            image_tensor,
            SPATIAL_QUERIES,
            threshold=0.165,
            use_softmax=True,
            temperature=0.1,
            return_grids=False
        )
        
        return {
            "global": global_analysis,
            "spatial": spatial_analysis,
            "combined_context": self.formatter.generate_combined_context(
                global_analysis, spatial_analysis
            )
        }
