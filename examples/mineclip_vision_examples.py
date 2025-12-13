import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../submodules/MineCLIP"))

try:
    from mineclip import MineCLIP
except ImportError as e:
    print(f"Error importing MineCLIP: {e}")
    print("Make sure you have run: pip install -r submodules/MineCLIP/requirements.txt")
    sys.exit(1)


class MineCLIPVisionModule:
    def __init__(self, weights_path: str, device: str = "cpu", hidden_dim: int = 512):
        self.device = device
        self.model = MineCLIP(
            arch="vit_base_p16_fz.v2.t2",
            resolution=(160, 256),
            pool_type="attn.d2.nh8.glusw",
            image_feature_dim=512,
            mlp_adapter_spec="v0-2.t0",
            hidden_dim=hidden_dim,
        ).to(device)

        if os.path.exists(weights_path):
            self.model.load_ckpt(weights_path, strict=True)
            print(f"✓ Loaded MineCLIP weights from {weights_path}")
        else:
            print(f"⚠ Weights not found at {weights_path}")
            print("  Download from: https://github.com/MineDojo/MineCLIP")

        self.model.eval()
        self.transform = T.Compose(
            [
                T.Resize((160, 256)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def encode_image(self, image_or_path):
        if isinstance(image_or_path, str):
            image = Image.open(image_or_path).convert("RGB")
        else:
            image = image_or_path

        image_tensor = self.transform(image).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_video(image_tensor)

        return embedding.squeeze(0).cpu()

    def query_similarity(self, image_or_path, text_query: str) -> float:
        if isinstance(image_or_path, str):
            image = Image.open(image_or_path).convert("RGB")
        else:
            image = image_or_path

        image_tensor = self.transform(image).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_emb = self.model.encode_video(image_tensor)
            similarity = torch.norm(image_emb, p=2).item()

        return similarity


def example_scene_understanding():
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Scene Understanding")
    print("=" * 70)

    weights_path = os.path.join(os.path.dirname(__file__), "../models/attn.pth")

    vision = MineCLIPVisionModule(weights_path=weights_path, device="mps")

    scene_queries = {
        "has_hostile_mobs": "skeleton creeper zombie nearby",
        "has_animals": "cow sheep pig chicken",
        "has_resources": "wood logs stone ore",
        "has_structures": "village house tower buildings",
        "weather": "rain thunderstorm raining",
        "time_of_day": "night dark sunset sunrise",
        "biome_type": "forest desert mountain plains jungle",
    }

    screenshot_path = "examples/mineclip_examples/images/example.png"

    if os.path.exists(screenshot_path):
        print(f"\nAnalyzing: {screenshot_path}\n")

        for aspect, query in scene_queries.items():
            score = vision.query_similarity(screenshot_path, query)
            print(f"  {aspect:20} | Score: {score:.3f} | Query: {query}")
    else:
        print(f"Screenshot not found at {screenshot_path}")
        print("Showing query examples for reference:")
        for aspect, query in scene_queries.items():
            print(f"  {aspect:20} | {query}")


def example_goal_aware_planning():
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Goal-Aware Vision for Planning")
    print("=" * 70)

    weights_path = os.path.join(os.path.dirname(__file__), "../models/attn.pth")

    vision = MineCLIPVisionModule(weights_path=weights_path, device="mps")

    goal_queries = {
        "gather_wood": ["trees nearby", "forest biome", "wood blocks visible"],
        "hunt_animals": ["animals visible", "open field", "mobs in view"],
        "avoid_danger": ["hostile mobs", "dangerous creatures", "threats nearby"],
        "find_water": ["water body", "river stream", "ocean sea"],
        "seek_shelter": ["cave entrance", "shelter building", "enclosed space"],
    }

    print("\nGoal-specific vision queries:")
    for goal, queries in goal_queries.items():
        print(f"\n  Goal: {goal}")
        for query in queries:
            print(f"    - {query}")


def example_vision_based_reward():
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Vision-Based Reward Signals")
    print("=" * 70)

    weights_path = os.path.join(os.path.dirname(__file__), "../models/attn.pth")

    vision = MineCLIPVisionModule(weights_path=weights_path, device="mps")

    code_example = """
def compute_vision_reward(vision_module, frame, goal):
    frame_embedding = vision_module.encode_image(frame)
    goal_similarity = vision_module.query_similarity(frame, goal)
    progress_reward = goal_similarity
    
    danger_penalty = 0.0
    if vision_module.query_similarity(frame, "hostile mobs nearby") > threshold:
        danger_penalty = -0.5
    
    total_reward = progress_reward + danger_penalty
    return total_reward
"""
    print(code_example)


def example_state_for_llm():
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Visual State for LLM Prompts")
    print("=" * 70)

    weights_path = os.path.join(os.path.dirname(__file__), "../models/attn.pth")

    vision = MineCLIPVisionModule(weights_path=weights_path, device="mps")

    llm_prompt_template = """
Current Visual Context:
- Environment: [BIOME_TYPE] biome
- Time: [TIME_OF_DAY]
- Threats: [THREAT_ANALYSIS]
- Resources: [VISIBLE_RESOURCES]
- Structures: [NEARBY_STRUCTURES]

Current Goal: [GOAL]

Decision: Based on the visual context above, what should I do next?

Previous Actions: [ACTION_HISTORY]

Your action:
"""
    print(llm_prompt_template)


def example_scene_clustering():
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Scene Clustering via Embeddings")
    print("=" * 70)

    weights_path = os.path.join(os.path.dirname(__file__), "../models/attn.pth")

    vision = MineCLIPVisionModule(weights_path=weights_path, device="mps")

    code_example = """
class SceneMemory:
    def __init__(self, vision_module):
        self.vision = vision_module
        self.scenes = []
    
    def add_scene(self, frame, metadata=None):
        emb = self.vision.encode_image(frame)
        self.scenes.append((emb, metadata))
    
    def find_similar(self, query_frame, k=5):
        query_emb = self.vision.encode_image(query_frame)
        similarities = [
            torch.nn.functional.cosine_similarity(
                query_emb.unsqueeze(0),
                scene_emb.unsqueeze(0)
            ).item()
            for scene_emb, _ in self.scenes
        ]
        top_k_indices = np.argsort(similarities)[-k:]
        return [self.scenes[i] for i in top_k_indices]
"""
    print(code_example)


def example_skill_selection():
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Vision-Guided Skill Selection")
    print("=" * 70)

    code_example = """
class SkillRouter:
    def __init__(self, vision_module):
        self.vision = vision_module
        
        self.skill_signatures = {
            "gather_wood": "forest trees wood blocks",
            "hunt_animals": "open field animals cows sheep",
            "find_shelter": "cave enclosed space dark",
            "get_water": "water body river stream",
            "mine": "stone ore cave underground",
        }
    
    def select_skill(self, frame):
        scores = {}
        for skill, description in self.skill_signatures.items():
            score = self.vision.query_similarity(frame, description)
            scores[skill] = score
        
        best_skill = max(scores, key=scores.get)
        return best_skill
"""
    print(code_example)


def example_integration_pattern():
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Integration Pattern")
    print("=" * 70)

    integration_code = """
from mineclip_vision_examples import MineCLIPVisionModule

class GameStateWithVision:
    def __init__(self, raw_obs, vision_module):
        self.raw_obs = raw_obs
        self.screenshot = raw_obs['screenshot']
        self.vision = vision_module
        self._analyze_scene()
    
    def _analyze_scene(self):
        self.scene_analysis = {
            "has_mobs": self.vision.query_similarity(self.screenshot, "hostile mobs nearby"),
            "has_animals": self.vision.query_similarity(self.screenshot, "passive animals"),
            "has_resources": self.vision.query_similarity(self.screenshot, "trees ore stone resources"),
            "is_safe": self.vision.query_similarity(self.screenshot, "peaceful safe location"),
        }
        self.embedding = self.vision.encode_image(self.screenshot)

class AgentWithVision:
    def __init__(self):
        self.vision = MineCLIPVisionModule(weights_path="weights/mineclip.pth", device="mps")
        self.planner = Planner()
    
    def step(self, raw_obs):
        state = GameStateWithVision(raw_obs, self.vision)
        goal = self.planner.plan(state)
        return goal
"""
    print(integration_code)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MineCLIP Usage Examples for Minecraft Agent Vision")
    print("=" * 70)

    example_scene_understanding()
    example_goal_aware_planning()
    example_vision_based_reward()
    example_state_for_llm()
    example_scene_clustering()
    example_skill_selection()
    example_integration_pattern()

    print("\n" + "=" * 70)
    print("Examples Complete")
    print("=" * 70)
