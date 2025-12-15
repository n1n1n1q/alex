"""
Image and text encoding utilities for MineCLIP vision module.
"""

import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

MINECLIP_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../submodules/MineCLIP")
)
if MINECLIP_PATH not in sys.path:
    sys.path.insert(0, MINECLIP_PATH)

from mineclip import MineCLIP
from mineclip.mineclip.tokenization import tokenize_batch


class MineCLIPEncoder:
    """
    Handles image and text encoding using MineCLIP model.
    """

    def __init__(self, model: MineCLIP, device: str = "cuda"):
        """
        Initialize encoder with MineCLIP model.

        Args:
            model: Initialized MineCLIP model
            device: Device to run on ('cpu', 'cuda', 'mps')
        """
        self.model = model
        self.device = device

        self.transform = T.Compose(
            [
                T.Resize((160, 256)),
            ]
        )

    def encode_image(self, image: Image.Image) -> torch.Tensor:
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

        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()

        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_video(image_tensor)
        return embedding

    def encode_text(self, text: str) -> torch.Tensor:
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

    def encode_text_batch(self, texts: list[str]) -> torch.Tensor:
        """
        Encode multiple text queries in batch.

        Args:
            texts: List of text descriptions

        Returns:
            Text embeddings tensor [N, dim]
        """
        all_tokens = tokenize_batch(texts, max_length=77, language_model="clip")
        all_tokens = all_tokens.to(self.device)
        with torch.no_grad():
            text_embs = self.model.encode_text(all_tokens)
        return text_embs

    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """
        Compute similarity between an image and text description.

        Args:
            image: PIL Image
            text: Text description

        Returns:
            Similarity score (higher = more similar)
        """
        image_emb = self.encode_image(image)
        text_emb = self.encode_text(text)

        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        logit_scale = self.model.clip_model.logit_scale.exp()
        similarity = (logit_scale * (image_emb @ text_emb.T)).squeeze().item()
        return similarity

    def compute_similarities_batch(
        self, image: Image.Image, texts: list[str]
    ) -> list[float]:
        """
        Compute similarities for multiple text queries efficiently.

        Args:
            image: PIL Image
            texts: List of text descriptions

        Returns:
            List of similarity scores
        """
        image_emb = self.encode_image(image)
        image_emb = F.normalize(image_emb, dim=-1)

        text_embs = self.encode_text_batch(texts)
        text_embs = F.normalize(text_embs, dim=-1)

        logit_scale = self.model.clip_model.logit_scale.exp()
        similarities = (logit_scale * (image_emb @ text_embs.T)).squeeze()

        return similarities.cpu().tolist()

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Get the raw embedding vector for an image.

        Args:
            image: PIL Image

        Returns:
            Numpy array of embedding
        """
        image_emb = self.encode_image(image)
        return image_emb.squeeze().cpu().numpy()

    def preprocess_image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image to tensor format for spatial analysis.

        Args:
            image: PIL Image

        Returns:
            Image tensor [1, C, H, W]
        """
        image = self.transform(image)
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return image_tensor
