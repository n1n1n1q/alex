"""
Spatial attention analysis for MineCLIP.
Divides the image into a semantic grid and provides spatial location information
for detected objects/features.
"""

import os
import sys
from typing import Dict

import torch
import torch.nn.functional as F

# Add MineCLIP to path
MINECLIP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../submodules/MineCLIP'))
if MINECLIP_PATH not in sys.path:
    sys.path.insert(0, MINECLIP_PATH)

from mineclip import MineCLIP
from mineclip.mineclip.tokenization import tokenize_batch


class SpatialAttentionMap:
    """
    Spatial attention analysis for MineCLIP.
    Divides the image into a semantic grid and provides spatial location information
    for detected objects/features.
    """
    
    def __init__(self, mineclip_model: MineCLIP, device: str = "cuda"):
        self.model = mineclip_model
        self.device = device
        
        # Image dimensions for MineCLIP
        self.img_height = 160
        self.img_width = 256
        self.patch_size = 16
        
        # Resulting patch grid dimensions
        self.grid_rows = self.img_height // self.patch_size  # 10
        self.grid_cols = self.img_width // self.patch_size   # 16
        
        # 4x4 Grid mapping for LLM
        self.semantic_grid_size = 4
        
        # Depth zones (row ranges in 10x16 grid)
        self.depth_zones = {
            "Sky/Ceiling": (0, 2),    # Top rows 0-2
            "Horizon/Far": (3, 5),    # Middle-top rows 3-5
            "Ground/Mid": (6, 7),     # Middle-bottom rows 6-7
            "Feet/Close": (8, 9)      # Bottom rows 8-9
        }
        
        # Horizontal zones (column ranges in 10x16 grid)
        self.horizontal_zones = {
            "Left": (0, 3),
            "Center-Left": (4, 7),
            "Center-Right": (8, 11),
            "Right": (12, 15)
        }
    
    def get_patch_embeddings(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level embeddings from the vision transformer.
        
        Args:
            image_tensor: Preprocessed image tensor [B, C, H, W]
            
        Returns:
            Patch embeddings [B, num_patches, dim]
        """
        with torch.no_grad():
            # Preprocess the image with MineCLIP normalization
            MC_IMAGE_MEAN = (0.3331, 0.3245, 0.3051)
            MC_IMAGE_STD = (0.2439, 0.2493, 0.2873)
            
            # Normalize
            mean = torch.tensor(MC_IMAGE_MEAN).view(1, 3, 1, 1).to(image_tensor.device)
            std = torch.tensor(MC_IMAGE_STD).view(1, 3, 1, 1).to(image_tensor.device)
            image_normalized = (image_tensor - mean) / std
            
            # Access the vision transformer (image encoder)
            vision_model = self.model.image_encoder
            
            # Forward through patch embedding layer
            x = vision_model.conv1(image_normalized)  # [batch, width, grid_h, grid_w]
            B = x.size(0)
            x = x.reshape(B, x.shape[1], -1)  # [batch, width, grid_h * grid_w]
            x = x.permute(0, 2, 1)  # [batch, num_patches, width]
            
            # Add CLS token and positional embeddings
            x = torch.cat(
                [vision_model.cls_token.repeat((B, 1, 1)), x], dim=1
            )  # [batch, num_patches + 1, width]
            x = x + vision_model.pos_embed
            
            # Pass through transformer blocks
            x = vision_model.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND (for transformer)
            x = vision_model.blocks(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            
            # Apply layer norm
            x = vision_model.ln_post(x)
            
            # Remove CLS token (index 0) - we only want spatial patches
            patch_embeddings = x[:, 1:, :]  # [batch, num_patches, width]
            
            # Project to output dimension (512) to match text embedding space
            if vision_model.projection is not None:
                patch_embeddings = patch_embeddings @ vision_model.projection
            
            return patch_embeddings
    
    def compute_similarity_maps(
        self, 
        patch_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        use_softmax: bool = True,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Compute spatial similarity maps between patches and text queries.
        
        Args:
            patch_embeddings: [B, num_patches, dim]
            text_embeddings: [Q, dim] where Q is number of queries
            use_softmax: Whether to apply softmax for competitive detection
            temperature: Temperature for softmax
            
        Returns:
            Similarity maps [B, Q, grid_rows, grid_cols]
        """
        batch_size = patch_embeddings.shape[0]
        
        # Reshape to spatial grid
        grid_tokens = patch_embeddings.reshape(
            batch_size, self.grid_rows, self.grid_cols, -1
        )  # [batch, 10, 16, dim]
        
        # Normalize embeddings
        grid_tokens_norm = F.normalize(grid_tokens, dim=-1)
        text_embeddings_norm = F.normalize(text_embeddings, dim=-1)
        
        # Compute similarity
        similarity_maps = torch.einsum(
            'brhd,qd->bqrh', 
            grid_tokens_norm, 
            text_embeddings_norm
        )
        
        if use_softmax:
            similarity_maps = F.softmax(
                similarity_maps / temperature, dim=1
            )
        
        return similarity_maps
    
    def downsample_to_4x4(self, similarity_map: torch.Tensor) -> torch.Tensor:
        """
        Downsample a 10x16 similarity map to a 4x4 semantic grid.
        
        Args:
            similarity_map: [10, 16] tensor
            
        Returns:
            4x4 grid tensor
        """
        grid_4x4 = torch.zeros(4, 4, device=similarity_map.device)
        
        # Define ranges for 4x4 grid
        depth_ranges = [
            (0, 3),   # Sky/Ceiling
            (3, 6),   # Horizon/Far
            (6, 8),   # Ground/Mid
            (8, 10)   # Feet/Close
        ]
        
        horiz_ranges = [
            (0, 4),    # Left
            (4, 8),    # Center-Left
            (8, 12),   # Center-Right
            (12, 16)   # Right
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
        top_k: int = 3
    ) -> str:
        """
        Generate a natural language description of spatial detections.
        
        Args:
            similarity_maps: Dict of query -> 4x4 grid
            threshold: Minimum confidence for detection
            top_k: Maximum number of detections to report
            
        Returns:
            Natural language description
        """
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
                depth_short = depth.split('/')[1] if '/' in depth else depth
                
                scored_detections.append((
                    max_score,
                    f"{query} detected in {horizontal} ({depth_short})"
                ))
        
        scored_detections.sort(reverse=True)
        detections = [desc for _, desc in scored_detections[:top_k]]
        
        if not detections:
            return "Visual Scan: No significant objects detected. Environment appears clear."
        
        description = "Visual Scan: " + ". ".join(detections) + "."
        return description
    
    def analyze_spatial(
        self,
        image_tensor: torch.Tensor,
        text_queries: list[str],
        threshold: float = 0.165,
        use_softmax: bool = True,
        temperature: float = 0.1,
        return_grids: bool = False
    ) -> dict:
        """
        Perform spatial analysis on an image.
        
        Args:
            image_tensor: Preprocessed image [B, C, H, W]
            text_queries: List of spatial queries
            threshold: Detection threshold
            use_softmax: Use competitive softmax detection
            temperature: Softmax temperature
            return_grids: Whether to return detailed grids
            
        Returns:
            Dictionary with spatial analysis results
        """
        # Get patch embeddings
        patch_embeddings = self.get_patch_embeddings(image_tensor)
        
        # Encode text queries
        with torch.no_grad():
            text_tokens = tokenize_batch(
                text_queries, max_length=77, language_model="clip"
            ).to(self.device)
            text_embeddings = self.model.encode_text(text_tokens)
        
        # Compute similarity maps
        similarity_maps = self.compute_similarity_maps(
            patch_embeddings, 
            text_embeddings,
            use_softmax=use_softmax,
            temperature=temperature
        )
        
        # Downsample to 4x4 grids
        similarity_maps = similarity_maps[0]  # Remove batch dim
        grids_4x4 = {}
        for i, query in enumerate(text_queries):
            grids_4x4[query] = self.downsample_to_4x4(similarity_maps[i])
        
        # Generate description
        description = self.generate_spatial_description(grids_4x4, threshold, top_k=3)
        
        result = {
            "description": description,
            "detections": self._extract_detections(grids_4x4, threshold),
        }
        
        if return_grids:
            result["grids_4x4"] = grids_4x4
            result["raw_maps"] = similarity_maps
        
        return result
    
    def _extract_detections(
        self, 
        grids_4x4: Dict[str, torch.Tensor], 
        threshold: float
    ) -> list[dict]:
        """
        Extract structured detection information from grids.
        
        Args:
            grids_4x4: Dict of query -> 4x4 grid
            threshold: Detection threshold
            
        Returns:
            List of detection dictionaries
        """
        depth_labels = ["Sky/Ceiling", "Horizon/Far", "Ground/Mid", "Feet/Close"]
        horiz_labels = ["Left", "Center-Left", "Center-Right", "Right"]
        
        detections = []
        for query, grid in grids_4x4.items():
            max_idx = grid.argmax()
            max_row = max_idx // 4
            max_col = max_idx % 4
            max_score = grid[max_row, max_col].item()
            
            if max_score > threshold:
                detections.append({
                    "object": query,
                    "confidence": round(max_score, 4),
                    "depth_zone": depth_labels[max_row],
                    "horizontal_zone": horiz_labels[max_col],
                })
        
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return detections
