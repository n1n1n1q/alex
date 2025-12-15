import torch
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import sys
import os
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../submodules/MineCLIP"))

try:
    from mineclip import MineCLIP
    from mineclip.mineclip.tokenization import tokenize_batch
except ImportError as e:
    print(f"Error importing MineCLIP: {e}")
    print("Make sure MineCLIP submodule is initialized and requirements are installed:")
    print("  cd alex/submodules/MineCLIP")
    print("  pip install -r requirements.txt")
    sys.exit(1)


class SpatialAttentionMap:
    def __init__(self, mineclip_model: MineCLIP, device: str = "cuda"):
        self.model = mineclip_model
        self.device = device

        self.img_height = 160
        self.img_width = 256
        self.patch_size = 16

        self.transform = T.Compose(
            [
                T.Resize((self.img_height, self.img_width)),
                T.ToTensor(),
            ]
        )

        self.grid_rows = self.img_height // self.patch_size
        self.grid_cols = self.img_width // self.patch_size

        self.semantic_grid_size = 4

        self.depth_zones = {
            "Sky/Ceiling": (0, 2),
            "Horizon/Far": (3, 5),
            "Ground/Mid": (6, 7),
            "Feet/Close": (8, 9),
        }

        self.horizontal_zones = {
            "Left": (0, 3),
            "Center-Left": (4, 7),
            "Center-Right": (8, 11),
            "Right": (12, 15),
        }

    def get_patch_embeddings(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            from mineclip.utils import basic_image_tensor_preprocess

            MC_IMAGE_MEAN = (0.3331, 0.3245, 0.3051)
            MC_IMAGE_STD = (0.2439, 0.2493, 0.2873)

            image = basic_image_tensor_preprocess(
                image, mean=MC_IMAGE_MEAN, std=MC_IMAGE_STD
            )

            vision_model = self.model.image_encoder

            x = vision_model.conv1(image)
            B = x.size(0)
            x = x.reshape(B, x.shape[1], -1)
            x = x.permute(0, 2, 1)

            # ...existing code...
            x = torch.cat(
                [vision_model.cls_token.repeat((B, 1, 1)), x], dim=1
            )
            x = x + vision_model.pos_embed

            x = vision_model.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = vision_model.blocks(x)
            x = x.permute(1, 0, 2)

            # ...existing code...
            x = vision_model.ln_post(x)

            patch_embeddings = x[:, 1:, :]

            # ...existing code...
            if vision_model.projection is not None:
                patch_embeddings = patch_embeddings @ vision_model.projection

            return patch_embeddings

    def compute_similarity_maps(
        self,
        patch_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        use_softmax: bool = True,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        batch_size = patch_embeddings.shape[0]

        grid_tokens = patch_embeddings.reshape(
            batch_size, self.grid_rows, self.grid_cols, -1
        )  # [batch, 10, 16, dim]

        grid_tokens_norm = torch.nn.functional.normalize(grid_tokens, dim=-1)
        text_embeddings_norm = torch.nn.functional.normalize(text_embeddings, dim=-1)

        similarity_maps = torch.einsum(
            "brhd,qd->bqrh", grid_tokens_norm, text_embeddings_norm
        )

        if use_softmax:
            similarity_maps = torch.nn.functional.softmax(
                similarity_maps / temperature, dim=1
            )

        return similarity_maps

    def downsample_to_4x4(self, similarity_map: torch.Tensor) -> torch.Tensor:
        grid_4x4 = torch.zeros(4, 4, device=similarity_map.device)

        depth_ranges = [
            (0, 3),
            (3, 6),
            (6, 8),
            (8, 10),
        ]

        horiz_ranges = [
            (0, 4),
            (4, 8),
            (8, 12),
            (12, 16),
        ]

        for i, (row_start, row_end) in enumerate(depth_ranges):
            for j, (col_start, col_end) in enumerate(horiz_ranges):
                region = similarity_map[row_start:row_end, col_start:col_end]
                grid_4x4[i, j] = region.mean()

        return grid_4x4

    def generate_spatial_description(
        self,
        similarity_maps: Dict[str, torch.Tensor],
        threshold: float = 0.3,
        top_k: int = 3,
    ) -> str:
        detections = []

        depth_labels = ["Sky/Ceiling", "Horizon/Far", "Ground/Mid", "Feet/Close"]
        horiz_labels = ["Left", "Center-Left", "Center-Right", "Right"]

        scored_detections = []
        for query, grid_4x4 in similarity_maps.items():
            max_idx = grid_4x4.argmax()
            max_row = max_idx // 4
            max_col = max_idx % 4
            max_score = grid_4x4[max_row, max_col].item()

            if max_score > threshold:
                depth = depth_labels[max_row]
                horizontal = horiz_labels[max_col]
                depth_short = depth.split("/")[1] if "/" in depth else depth

                scored_detections.append(
                    (max_score, f"{query} detected in {horizontal} ({depth_short})")
                )

        scored_detections.sort(reverse=True)
        detections = [desc for _, desc in scored_detections[:top_k]]

        if not detections:
            return "Visual Scan: No significant objects detected. Environment appears clear."

        description = "Visual Scan: " + ". ".join(detections) + "."
        return description

    def load_image(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img_tensor

    def analyze_frame(
        self,
        image: torch.Tensor,
        text_queries: List[str],
        threshold: float = 0.3,
        return_grids: bool = False,
        use_softmax: bool = True,
        temperature: float = 0.1,
        global_queries: List[str] = None,
    ) -> Dict:
        patch_embeddings = self.get_patch_embeddings(image)

        with torch.no_grad():
            text_tokens = tokenize_batch(
                text_queries, max_length=77, language_model="clip"
            ).to(self.device)
            text_embeddings = self.model.encode_text(text_tokens)

        similarity_maps = self.compute_similarity_maps(
            patch_embeddings,
            text_embeddings,
            use_softmax=use_softmax,
            temperature=temperature,
        )

        global_scores = {}
        if global_queries:
            with torch.no_grad():
                image_features = self.model.forward_image_features(image.unsqueeze(1))
                video_features = self.model.forward_video_features(image_features)

                global_tokens = tokenize_batch(
                    global_queries, max_length=77, language_model="clip"
                ).to(self.device)
                global_text_embeddings = self.model.encode_text(global_tokens)

                video_features_norm = torch.nn.functional.normalize(
                    video_features, dim=-1
                )
                global_text_norm = torch.nn.functional.normalize(
                    global_text_embeddings, dim=-1
                )

                similarities = torch.mm(video_features_norm, global_text_norm.t())

                for i, query in enumerate(global_queries):
                    global_scores[query] = similarities[0, i].item()

        similarity_maps = similarity_maps[0]
        grids_4x4 = {}
        for i, query in enumerate(text_queries):
            grids_4x4[query] = self.downsample_to_4x4(similarity_maps[i])

        description = self.generate_spatial_description(grids_4x4, threshold, top_k=3)

        result = {
            "description": description,
            "global_scores": global_scores if global_queries else None,
        }

        if return_grids:
            result["grids_4x4"] = grids_4x4
            result["raw_maps"] = similarity_maps

        return result


def analyze_image(
    spatial_system: SpatialAttentionMap,
    image_path: str,
    text_queries: List[str],
    use_softmax: bool = True,
    global_queries: List[str] = None,
):
    """
    Analyze a single image and display results.
    """
    print("\n" + "=" * 80)
    print(f"Analyzing: {Path(image_path).name}")
    print("=" * 80)

    image = spatial_system.load_image(image_path)

    mode = "Softmax (Competitive)" if use_softmax else "Raw Cosine Similarity"
    print(f"Detection Mode: {mode}")
    print(f"Spatial queries: {', '.join(text_queries)}")
    if global_queries:
        print(f"Global queries: {', '.join(global_queries)}")

    result = spatial_system.analyze_frame(
        image,
        text_queries,
        threshold=0.165 if use_softmax else 0.26,
        return_grids=True,
        use_softmax=use_softmax,
        temperature=0.1,
        global_queries=global_queries,
    )

    print("\n" + "-" * 80)
    print("Natural Language Output (for LLM):")
    print("-" * 80)
    print(result["description"])

    if result.get("global_scores"):
        print("\n" + "-" * 80)
        print("Global Scene Attributes:")
        print("-" * 80)
        sorted_globals = sorted(
            result["global_scores"].items(), key=lambda x: x[1], reverse=True
        )
        for query, score in sorted_globals:
            print(f"  {score:.4f} - '{query}'")

    if "grids_4x4" in result:
        print("\n" + "-" * 80)
        print("Top Detections by Zone:")
        print("-" * 80)

        depth_labels = ["Sky/Ceiling", "Horizon/Far", "Ground/Mid", "Feet/Close"]
        horiz_labels = ["Left", "Center-Left", "Center-Right", "Right"]

        detections = []
        for query, grid in result["grids_4x4"].items():
            max_val = grid.max().item()
            max_idx = grid.argmax()
            max_row = max_idx // 4
            max_col = max_idx % 4
            detections.append(
                (max_val, query, depth_labels[max_row], horiz_labels[max_col])
            )

        detections.sort(reverse=True)

        for conf, query, depth, horiz in detections[:5]:
            print(f"  {conf:.4f} - '{query}' at {horiz}, {depth}")

    return result


def main():
    """
    Example usage of the Spatial Attention Map system with real Minecraft screenshots.
    """
    print("MineCLIP Spatial Attention Map Example - 'The Semantic Retina'")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\nLoading MineCLIP model...")
    model_path = Path(__file__).parent.parent / "models" / "attn.pth"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please download the MineCLIP model first.")
        print("Expected location: alex/models/attn.pth")
        return

    mineclip = MineCLIP(
        arch="vit_base_p16_fz.v2.t2",
        resolution=(160, 256),
        pool_type="attn.d2.nh8.glusw",
        image_feature_dim=512,
        mlp_adapter_spec="v0-2.t0",
        hidden_dim=512,
    ).to(device)

    mineclip.load_ckpt(str(model_path), strict=True)
    mineclip.eval()

    print("✓ Model loaded successfully")

    # ...existing code...
    spatial_system = SpatialAttentionMap(mineclip, device=device)

    print(f"\nSpatial Grid Configuration:")
    print(f"  Input Image: {spatial_system.img_width}x{spatial_system.img_height}")
    print(f"  Patch Size: {spatial_system.patch_size}x{spatial_system.patch_size}")
    print(f"  Patch Grid: {spatial_system.grid_cols}x{spatial_system.grid_rows}")
    print(f"  Semantic Grid: 4x4")

    images_dir = Path(__file__).parent / "mineclip_examples" / "images"

    if not images_dir.exists():
        print(f"\nWarning: Images directory not found at {images_dir}")
        print("Using synthetic test image instead...")
        dummy_image = torch.randn(1, 3, 160, 256).to(device)
        text_queries = [
            "a creeper",
            "iron ore",
            "wooden planks",
            "stone blocks",
            "grass",
            "trees",
        ]
        result = spatial_system.analyze_frame(
            dummy_image, text_queries, threshold=0.25, return_grids=True
        )
        print("\nNatural Language Output:")
        print(result["description"])
        return

    # ...existing code...
    image_files = list(images_dir.glob("*.png"))

    # ...existing code...
    # ...existing code...
    # ...existing code...
    queries_by_scene = {
        "zombie": {
            "spatial": ["zombie", "skeleton", "creeper", "player hand", "weapon"],
            "global": [
                "hostile environment",
                "night time",
                "dangerous situation",
                "combat",
                "dark",
            ],
        },
        "nether": {
            "spatial": ["lava", "netherrack", "ghast", "fire", "fortress"],
            "global": [
                "nether dimension",
                "hell biome",
                "red landscape",
                "dangerous",
                "hot environment",
            ],
        },
        "taiga": {
            "spatial": ["spruce trees", "snow patches", "ice", "animals"],
            "global": [
                "snowy biome",
                "cold environment",
                "winter forest",
                "taiga",
                "peaceful",
            ],
        },
        "chicken": {
            "spatial": ["chicken", "animals", "grass", "fence"],
            "global": ["farm", "peaceful", "daylight", "animal pen", "safe area"],
        },
        "indoor": {
            "spatial": ["walls", "ceiling", "torches", "crafting table", "chest"],
            "global": [
                "enclosed space",
                "building interior",
                "shelter",
                "protected area",
                "base",
            ],
        },
        "default": {
            "spatial": ["trees", "grass", "stone", "wood", "animals", "mobs"],
            "global": [
                "overworld",
                "natural terrain",
                "wilderness",
                "outdoor",
                "minecraft world",
            ],
        },
    }

    # ...existing code...
    zombie_images = [img for img in image_files if "zombie" in img.stem.lower()]

    if zombie_images:
        zombie_path = zombie_images[0]
        scene_queries = queries_by_scene["zombie"]

        print("\n" + "=" * 80)
        print("HYBRID DETECTION: Spatial + Global Queries")
        print("=" * 80)
        print("\nSpatial queries detect WHERE objects are (per 4x4 grid)")
        print("Global queries detect WHAT the overall scene is (whole image)")

        analyze_image(
            spatial_system,
            str(zombie_path),
            scene_queries["spatial"],
            use_softmax=True,
            global_queries=scene_queries["global"],
        )
    else:
        print("\nNo zombie image found. Analyzing first 3 images instead...")
        for img_path in image_files[:3]:
            img_name = img_path.stem.lower()
            scene_queries = queries_by_scene.get(img_name, queries_by_scene["default"])
            analyze_image(
                spatial_system,
                str(img_path),
                scene_queries["spatial"],
                use_softmax=True,
                global_queries=scene_queries["global"],
            )

    # ...existing code...
    print("\n" + "=" * 80)
    print("✓ Analysis completed successfully!")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Connect to your LLM planner for action generation")
    print("2. Integrate with Steve-1 executor for command execution")
    print("3. Implement temporal consistency across frames")
    print("4. Add object tracking across multiple frames")
    print("\nUsage:")
    print("  from mineclip_spatial_attention import SpatialAttentionMap")
    print("  result = spatial_system.analyze_frame(image, queries)")
    print("  llm_input = result['description']")


if __name__ == "__main__":
    main()
