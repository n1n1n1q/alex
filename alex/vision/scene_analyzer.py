"""
Scene analysis utilities for categorizing and analyzing Minecraft environments.
"""

from typing import Dict
import torch
import torch.nn.functional as F
from PIL import Image

from .encoders import MineCLIPEncoder

try:
    from .vision_queries import SCENE_QUERIES
except ImportError:
    from alex.vision.vision_queries import SCENE_QUERIES


class SceneAnalyzer:
    """
    Analyzes Minecraft scenes across multiple categories using MineCLIP.
    """

    def __init__(self, encoder: MineCLIPEncoder):
        """
        Initialize scene analyzer.

        Args:
            encoder: MineCLIPEncoder instance for encoding operations
        """
        self.encoder = encoder

    def analyze_category(self, category: str, image: Image.Image) -> dict:
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
                "probabilities": {},
            }

        names = list(queries.keys())
        texts = list(queries.values())

        raw_scores = self.encoder.compute_similarities_batch(image, texts)

        scores = {name: round(score, 4) for name, score in zip(names, raw_scores)}

        scores_tensor = torch.tensor(raw_scores)
        probs = F.softmax(scores_tensor, dim=0).tolist()
        probabilities = {name: round(prob, 4) for name, prob in zip(names, probs)}

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

    def analyze_comprehensive(self, image: Image.Image) -> dict:
        """
        Perform comprehensive analysis of a Minecraft screenshot.

        Args:
            image: PIL Image to analyze

        Returns:
            Dictionary with comprehensive scene analysis
        """
        embedding = self.encoder.get_embedding(image)

        biome_analysis = self.analyze_category("biome", image)
        time_analysis = self.analyze_category("time_of_day", image)
        weather_analysis = self.analyze_category("weather", image)
        safety_analysis = self.analyze_category("safety", image)
        hostile_analysis = self.analyze_category("hostile_mobs", image)
        passive_analysis = self.analyze_category("passive_mobs", image)
        resources_analysis = self.analyze_category("resources", image)
        structures_analysis = self.analyze_category("structures", image)

        analysis = {
            "summary": {
                "biome": biome_analysis["best_match"],
                "time": time_analysis["best_match"],
                "weather": weather_analysis["best_match"],
                "safety": safety_analysis["best_match"],
                "top_threat": (
                    hostile_analysis["best_match"]
                    if hostile_analysis["confidence"] > 0.15
                    else None
                ),
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
        image = Image.open(image_path).convert("RGB")
        analysis = self.analyze_comprehensive(image)
        analysis["image_path"] = image_path
        return analysis
