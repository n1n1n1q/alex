import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../submodules/MineCLIP'))
from mineclip import MineCLIP
from mineclip.mineclip.tokenization import tokenize_batch


MC_IMAGE_MEAN = (0.3331, 0.3245, 0.3051)
MC_IMAGE_STD = (0.2439, 0.2493, 0.2873)


class MineCLIPAnalyzer:
    SCENE_QUERIES = {
        "biome": {
            "forest": "forest trees oak birch spruce",
            "plains": "flat grass plains open field",
            "desert": "desert sand cactus dry",
            "mountain": "mountain hills stone high elevation",
            "jungle": "jungle vines tropical dense trees",
            "snow": "snow ice cold frozen winter",
            "swamp": "swamp water lily pads murky",
            "ocean": "ocean water sea waves beach",
            "cave": "cave underground dark stone",
        },
        "time_of_day": {
            "day": "bright daylight sunny clear sky",
            "night": "dark night stars moon",
            "sunset": "sunset orange sky evening dusk",
            "sunrise": "sunrise morning dawn",
        },
        "weather": {
            "clear": "clear sky sunny",
            "rain": "rain raining wet weather",
            "storm": "thunderstorm lightning dark clouds",
        },
        "hostile_mobs": {
            "zombie": "zombie undead green monster",
            "skeleton": "skeleton archer bones bow",
            "creeper": "creeper green explosive monster",
            "spider": "spider black legs eight",
            "enderman": "enderman tall black purple eyes",
        },
        "passive_mobs": {
            "cow": "cow cattle brown animal",
            "sheep": "sheep wool white animal",
            "pig": "pig pink animal",
            "chicken": "chicken bird feathers",
            "horse": "horse mount riding",
        },
        "resources": {
            "wood": "trees logs wood oak birch",
            "stone": "stone rocks cobblestone",
            "ore": "ore mining iron coal diamond",
            "water": "water river lake pond",
            "crops": "wheat carrots potatoes farming",
        },
        "structures": {
            "village": "village houses buildings npcs",
            "house": "building house structure shelter",
            "cave_entrance": "cave entrance hole opening",
            "chest": "chest loot container",
        },
        "safety": {
            "safe": "peaceful calm safe area",
            "dangerous": "danger hostile threatening monsters",
            "enclosed": "enclosed protected walls ceiling",
            "exposed": "open exposed outside vulnerable",
        },
    }
    
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = device
        
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
            print(f"✓ Loaded MineCLIP weights: {weights_path}", file=sys.stderr)
        else:
            print(f"⚠ Weights not found: {weights_path}", file=sys.stderr)
            sys.exit(1)
        
        self.model.eval()
        
        self.transform = T.Compose([
            T.Resize((160, 256)),
            T.ToTensor(),
            T.Normalize(mean=MC_IMAGE_MEAN, std=MC_IMAGE_STD)
        ])
    
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        image_tensor = self.transform(image).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_video(image_tensor)
        return embedding
    
    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = tokenize_batch([text], max_length=77, language_model="clip")
        tokens = tokens.to(self.device)
        with torch.no_grad():
            text_emb = self.model.encode_text(tokens)
        return text_emb
    
    def compute_similarity(self, image: Image.Image, text: str) -> float:
        image_emb = self._encode_image(image)
        text_emb = self._encode_text(text)
        
        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)
        
        similarity = (image_emb @ text_emb.T).squeeze().item()
        return similarity
    
    def _analyze_category(self, category: str, image: Image.Image) -> dict:
        queries = self.SCENE_QUERIES.get(category, {})
        
        scores = {}
        for name, query_text in queries.items():
            scores[name] = round(self.compute_similarity(image, query_text), 4)
        
        if scores:
            best_match = max(scores, key=scores.get)
            best_score = scores[best_match]
        else:
            best_match = None
            best_score = 0.0
        
        return {
            "scores": scores,
            "best_match": best_match,
            "confidence": best_score,
        }
    
    def analyze(self, image_path: str) -> dict:
        image = Image.open(image_path).convert('RGB')
        image_emb = self._encode_image(image)
        embedding = image_emb.squeeze().cpu().numpy()
        
        biome_analysis = self._analyze_category("biome", image)
        time_analysis = self._analyze_category("time_of_day", image)
        weather_analysis = self._analyze_category("weather", image)
        safety_analysis = self._analyze_category("safety", image)
        hostile_analysis = self._analyze_category("hostile_mobs", image)
        passive_analysis = self._analyze_category("passive_mobs", image)
        resources_analysis = self._analyze_category("resources", image)
        structures_analysis = self._analyze_category("structures", image)
        
        analysis = {
            "image_path": image_path,
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
            "embedding_sample": embedding[:10].tolist(),
        }
        
        return analysis
    
    def generate_agent_context(self, image_path: str) -> str:
        analysis = self.analyze(image_path)
        summary = analysis["summary"]
        threat_info = summary["top_threat"] if summary["top_threat"] else "none detected"
        env = analysis["environment"]
        entities = analysis["entities"]
        
        context = f"""
=== MINECRAFT VISUAL ANALYSIS ===

QUICK SUMMARY:
  Biome: {summary['biome']} (confidence: {env['biome']['confidence']:.2f})
  Time: {summary['time']} (confidence: {env['time']['confidence']:.2f})
  Weather: {summary['weather']} (confidence: {env['weather']['confidence']:.2f})
  Safety: {summary['safety']} (confidence: {analysis['safety']['confidence']:.2f})
  Primary Threat: {threat_info}
  Top Resource: {summary['top_resource']}

DETAILED SCORES:

Environment/Biome:
{self._format_scores(env['biome']['scores'])}

Time of Day:
{self._format_scores(env['time']['scores'])}

Weather:
{self._format_scores(env['weather']['scores'])}

Safety Assessment:
{self._format_scores(analysis['safety']['scores'])}

Hostile Mobs (threat detection):
{self._format_scores(entities['hostile_mobs']['scores'])}

Passive Mobs:
{self._format_scores(entities['passive_mobs']['scores'])}

Resources:
{self._format_scores(analysis['resources']['scores'])}

Structures:
{self._format_scores(analysis['structures']['scores'])}

=== END ANALYSIS ===
"""
        return context
    
    def _format_scores(self, scores: dict) -> str:
        lines = []
        for name, score in sorted(scores.items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"  {name:15} [{bar}] {score:.3f}")
        return "\n".join(lines)
    
    def get_embedding(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert('RGB')
        image_emb = self._encode_image(image)
        return image_emb.squeeze().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Analyze Minecraft screenshots for agent decision-making")
    parser.add_argument("image", help="Path to Minecraft screenshot")
    parser.add_argument("--weights", default=None, help="Path to MineCLIP weights (default: models/attn.pth)")
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"], help="Device to run on")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of text")
    parser.add_argument("--embedding", action="store_true", help="Output just the embedding vector")
    
    args = parser.parse_args()
    
    if args.weights is None:
        args.weights = os.path.join(os.path.dirname(__file__), '../models/attn.pth')
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    analyzer = MineCLIPAnalyzer(weights_path=args.weights, device=args.device)
    
    if args.embedding:
        embedding = analyzer.get_embedding(args.image)
        if args.json:
            print(json.dumps({"embedding": embedding.tolist()}))
        else:
            print(f"Embedding shape: {embedding.shape}")
            print(f"Embedding: {embedding}")
    elif args.json:
        analysis = analyzer.analyze(args.image)
        print(json.dumps(analysis, indent=2))
    else:
        context = analyzer.generate_agent_context(args.image)
        print(context)


if __name__ == "__main__":
    main()
