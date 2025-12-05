from __future__ import annotations

import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Add MineCLIP to path
MINECLIP_PATH = os.path.join(os.path.dirname(__file__), '../../submodules/MineCLIP')
if MINECLIP_PATH not in sys.path:
    sys.path.insert(0, MINECLIP_PATH)

from mineclip import MineCLIP
from mineclip.mineclip.tokenization import tokenize_batch

from .vision_queries import SCENE_QUERIES


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
            weights_path = os.path.join(
                os.path.dirname(__file__), 
                '../../models/attn.pth'
            )
        
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
        
        # Image preprocessing transform
        self.transform = T.Compose([
            T.Resize((160, 256)),
        ])
    
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode a PIL image to MineCLIP embedding.
        
        MineCLIP expects input tensor with shape [B, L, C, H, W] where:
        - B = batch size
        - L = sequence length (number of frames, use 1 for single image)
        - C = channels (3 for RGB)
        - H, W = height, width (160, 256 for MineCLIP)
        
        Pixel values should be in [0, 255] range as the model handles normalization.
        
        Args:
            image: PIL Image in RGB format
            
        Returns:
            Image embedding tensor
        """
        image = self.transform(image)
        
        # Convert to tensor [C, H, W] with values in [0, 255]
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        
        # Add batch and sequence dimensions: [1, 1, C, H, W]
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_video(image_tensor)
        return embedding
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text to MineCLIP embedding.
        
        Args:
            text: Text description to encode
            
        Returns:
            Text embedding tensor
        """
        tokens = tokenize_batch([text], max_length=77, language_model="clip")
        tokens = tokens.to(self.device)
        with torch.no_grad():
            text_emb = self.model.encode_text(tokens)
        return text_emb
    
    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """
        Compute similarity between an image and text description.
        
        Args:
            image: PIL Image
            text: Text description
            
        Returns:
            Similarity score (higher = more similar)
        """
        image_emb = self._encode_image(image)
        text_emb = self._encode_text(text)
        
        # Normalize embeddings
        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)
        
        # Compute scaled similarity
        logit_scale = self.model.clip_model.logit_scale.exp()
        similarity = (logit_scale * (image_emb @ text_emb.T)).squeeze().item()
        return similarity
    
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
        image_emb = self._encode_image(image)
        image_emb = F.normalize(image_emb, dim=-1)
        
        # Encode all texts in batch
        all_tokens = tokenize_batch(texts, max_length=77, language_model="clip")
        all_tokens = all_tokens.to(self.device)
        with torch.no_grad():
            text_embs = self.model.encode_text(all_tokens)
        text_embs = F.normalize(text_embs, dim=-1)
        
        # Compute scaled similarities
        logit_scale = self.model.clip_model.logit_scale.exp()
        similarities = (logit_scale * (image_emb @ text_embs.T)).squeeze()
        
        return similarities.cpu().tolist()
    
    def _analyze_category(self, category: str, image: Image.Image) -> dict:
        """
        Analyze a specific category of scene features.
        
        Args:
            category: Category name from SCENE_QUERIES
            image: PIL Image to analyze
            
        Returns:
            Dictionary with scores, probabilities, and best match
        """
        queries = SCENE_QUERIES.get(category, {})
        
        if not queries:
            return {
                "scores": {}, 
                "best_match": None, 
                "confidence": 0.0, 
                "probabilities": {}
            }
        
        names = list(queries.keys())
        texts = list(queries.values())
        
        # Get raw similarity scores
        raw_scores = self.compute_similarities_batch(image, texts)
        
        # Create named scores
        scores = {name: round(score, 4) for name, score in zip(names, raw_scores)}
        
        # Compute probabilities using softmax
        scores_tensor = torch.tensor(raw_scores)
        probs = F.softmax(scores_tensor, dim=0).tolist()
        probabilities = {name: round(prob, 4) for name, prob in zip(names, probs)}
        
        # Find best match
        best_match = max(scores, key=scores.get)
        best_score = scores[best_match]
        best_prob = probabilities[best_match]
        
        return {
            "scores": scores,
            "probabilities": probabilities,
            "best_match": best_match,
            "confidence": best_score,
            "probability": best_prob,
        }
    
    def analyze_image(self, image: Image.Image) -> dict:
        """
        Perform comprehensive analysis of a Minecraft screenshot.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary with comprehensive scene analysis
        """
        # Get image embedding
        image_emb = self._encode_image(image)
        embedding = image_emb.squeeze().cpu().numpy()
        
        # Analyze all categories
        biome_analysis = self._analyze_category("biome", image)
        time_analysis = self._analyze_category("time_of_day", image)
        weather_analysis = self._analyze_category("weather", image)
        safety_analysis = self._analyze_category("safety", image)
        hostile_analysis = self._analyze_category("hostile_mobs", image)
        passive_analysis = self._analyze_category("passive_mobs", image)
        resources_analysis = self._analyze_category("resources", image)
        structures_analysis = self._analyze_category("structures", image)
        
        # Build comprehensive analysis
        analysis = {
            "summary": {
                "biome": biome_analysis["best_match"],
                "time": time_analysis["best_match"],
                "weather": weather_analysis["best_match"],
                "safety": safety_analysis["best_match"],
                "top_threat": hostile_analysis["best_match"] if hostile_analysis["confidence"] > 0.15 else None,
                "top_resource": resources_analysis["best_match"],
            },
            "environment": {
                "biome": biome_analysis,
                "time": time_analysis,
                "weather": weather_analysis,
            },
            "safety": safety_analysis,
            "entities": {
                "hostile_mobs": hostile_analysis,
                "passive_mobs": passive_analysis,
            },
            "resources": resources_analysis,
            "structures": structures_analysis,
            "embedding": embedding,
        }
        
        return analysis
    
    def analyze_from_path(self, image_path: str) -> dict:
        """
        Analyze an image from a file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with comprehensive scene analysis
        """
        image = Image.open(image_path).convert('RGB')
        analysis = self.analyze_image(image)
        analysis["image_path"] = image_path
        return analysis
    
    def generate_context_string(self, analysis: dict) -> str:
        """
        Generate a human-readable context string from analysis results.
        
        Args:
            analysis: Analysis dictionary from analyze_image()
            
        Returns:
            Formatted context string for agent
        """
        summary = analysis["summary"]
        threat_info = summary["top_threat"] if summary["top_threat"] else "none detected"
        env = analysis["environment"]
        entities = analysis["entities"]
        
        context = f"""=== MINECRAFT VISUAL ANALYSIS ===

QUICK SUMMARY:
  Biome: {summary['biome']} (probability: {env['biome']['probability']:.1%})
  Time: {summary['time']} (probability: {env['time']['probability']:.1%})
  Weather: {summary['weather']} (probability: {env['weather']['probability']:.1%})
  Safety: {summary['safety']} (probability: {analysis['safety']['probability']:.1%})
  Primary Threat: {threat_info}
  Top Resource: {summary['top_resource']}

DETAILED PROBABILITIES:

Environment/Biome:
{self._format_scores(env['biome']['probabilities'])}

Time of Day:
{self._format_scores(env['time']['probabilities'])}

Weather:
{self._format_scores(env['weather']['probabilities'])}

Safety Assessment:
{self._format_scores(analysis['safety']['probabilities'])}

Hostile Mobs (threat detection):
{self._format_scores(entities['hostile_mobs']['probabilities'])}

Passive Mobs:
{self._format_scores(entities['passive_mobs']['probabilities'])}

Resources:
{self._format_scores(analysis['resources']['probabilities'])}

Structures:
{self._format_scores(analysis['structures']['probabilities'])}

=== END ANALYSIS ===
"""
        return context
    
    def _format_scores(self, scores: dict, as_percent: bool = True) -> str:
        """
        Format scores as a visual bar chart.
        
        Args:
            scores: Dictionary of name -> score
            as_percent: Whether to format as percentage (for probabilities)
            
        Returns:
            Formatted string with bars
        """
        lines = []
        for name, score in sorted(scores.items(), key=lambda x: -x[1]):
            bar_len = int(score * 20) if score <= 1.0 else int(min(score / 5, 1.0) * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            if as_percent and score <= 1.0:
                lines.append(f"  {name:15} [{bar}] {score:.1%}")
            else:
                lines.append(f"  {name:15} [{bar}] {score:.2f}")
        return "\n".join(lines)
    
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Get the raw embedding vector for an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Numpy array of embedding
        """
        image_emb = self._encode_image(image)
        return image_emb.squeeze().cpu().numpy()
