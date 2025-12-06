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
    Analyzes game screenshots to provide environmental context.
    """
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the vision module with MineCLIP.
        
        Args:
            weights_path: Path to MineCLIP weights. If None, defaults to ../models/attn.pth
            device: Device to run on ('cpu', 'cuda', 'mps'). Auto-detects if None.
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        # Default weights path
        if weights_path is None:
            weights_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                '../models/avg.pth'
            ))
        else:
            # Ensure absolute path
            weights_path = os.path.abspath(weights_path)
        
        # Determine pool type from weights filename
        if "avg" in os.path.basename(weights_path).lower():
            pool_type = "avg"
        else:
            pool_type = "attn.d2.nh8.glusw"
        
        # Initialize MineCLIP model
        self.model = MineCLIP(
            arch="vit_base_p16_fz.v2.t2",
            resolution=(160, 256),
            pool_type=pool_type,
            image_feature_dim=512,
            mlp_adapter_spec="v0-2.t0",
            hidden_dim=512,
        ).to(device)
        
        # Load weights
        if os.path.exists(weights_path):
            self.model.load_ckpt(weights_path, strict=True)
            print(f"✓ Loaded MineCLIP weights: {weights_path}")
        else:
            print(f"⚠ Warning: Weights not found at {weights_path}")
            print(f"  Model initialized but not loaded. Vision may not work properly.")
        
        self.model.eval()
        
        # Initialize modular components
        self.encoder = MineCLIPEncoder(self.model, device)
        self.scene_analyzer = SceneAnalyzer(self.encoder)
        self.spatial_attention = SpatialAttentionMap(self.model, device)
        self.formatter = VisionFormatter()
    
    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """
        Compute similarity between an image and text description.
        
        Args:
            image: PIL Image
            text: Text description
            
        Returns:
            Similarity score (higher = more similar)
        """
        return self.encoder.compute_similarity(image, text)
    
    def compute_similarities_batch(
        self, 
        image: Image.Image, 
        texts: list[str]
    ) -> list[float]:
        """
        Compute similarities for multiple text queries efficiently.
        
        Args:
            image: PIL Image
            texts: List of text descriptions
            
        Returns:
            List of similarity scores
        """
        return self.encoder.compute_similarities_batch(image, texts)
    
    def analyze_image(self, image: Image.Image) -> dict:
        """
        Perform comprehensive analysis of a Minecraft screenshot.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary with comprehensive scene analysis
        """
        return self.scene_analyzer.analyze_comprehensive(image)
    
    def analyze_from_path(self, image_path: str) -> dict:
        """
        Analyze an image from a file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with comprehensive scene analysis
        """
        return self.scene_analyzer.analyze_from_path(image_path)
    
    def generate_context_string(self, analysis: dict) -> str:
        """
        Generate a human-readable context string from analysis results.
        
        Args:
            analysis: Analysis dictionary from analyze_image()
            
        Returns:
            Formatted context string for agent
        """
        return self.formatter.generate_context_string(analysis)
    
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Get the raw embedding vector for an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Numpy array of embedding
        """
        return self.encoder.get_embedding(image)
    
    def analyze_spatial(
        self,
        image: Image.Image,
        spatial_queries: Optional[list[str]] = None,
        threshold: float = 0.165,
        use_softmax: bool = True,
        temperature: float = 0.1,
        return_grids: bool = False
    ) -> dict:
        """
        Perform spatial analysis to detect WHERE objects are located.
        
        Args:
            image: PIL Image to analyze
            spatial_queries: List of objects to detect spatially. 
                           If None, uses default queries from vision_queries.py
            threshold: Detection confidence threshold (0-1)
            use_softmax: Use competitive softmax for detection (recommended)
            temperature: Softmax temperature for sharpness
            return_grids: Whether to return detailed 4x4 grids
            
        Returns:
            Dictionary with spatial analysis including:
            - description: Natural language description of detections
            - detections: List of structured detection info
            - grids_4x4: Optional 4x4 grids per query
        """
        from .vision_queries import SPATIAL_QUERIES
        
        if spatial_queries is None:
            spatial_queries = SPATIAL_QUERIES
        
        # Preprocess image to tensor
        image_tensor = self.encoder.preprocess_image_to_tensor(image)
        
        # Run spatial analysis
        result = self.spatial_attention.analyze_spatial(
            image_tensor,
            spatial_queries,
            threshold=threshold,
            use_softmax=use_softmax,
            temperature=temperature,
            return_grids=return_grids
        )
        
        return result
    
    def analyze_comprehensive(self, image: Image.Image) -> dict:
        """
        Perform both global scene analysis AND spatial analysis.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary with both global and spatial analysis
        """
        # Get global scene analysis
        global_analysis = self.analyze_image(image)
        
        # Get spatial analysis
        spatial_analysis = self.analyze_spatial(image)
        
        # Combine results
        return {
            "global": global_analysis,
            "spatial": spatial_analysis,
            "combined_context": self.formatter.generate_combined_context(
                global_analysis, spatial_analysis
            )
        }
