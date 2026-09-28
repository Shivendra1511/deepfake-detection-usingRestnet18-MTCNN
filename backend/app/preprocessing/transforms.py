from typing import Optional
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from app.config import settings
from app.preprocessing.face_crop import apply_clahe

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_inference_transforms() -> transforms.Compose:
    """
    Deterministic ImageNet normalization transform for inference and validation.
    Converts (H, W, C) PIL/ndarray in [0, 255] to (C, H, W) Tensor in [0, 1] then normalizes.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def preprocess_face_tensor(
    face_rgb: np.ndarray,
    use_clahe: Optional[bool] = None
) -> torch.Tensor:
    """
    Take a cropped 224x224 RGB numpy face, optionally apply CLAHE,
    and return normalized tensor of shape (3, 224, 224).
    """
    clahe_enabled = settings.USE_CLAHE if use_clahe is None else use_clahe
    if clahe_enabled:
        face_rgb = apply_clahe(
            face_rgb,
            clip_limit=settings.CLAHE_CLIP_LIMIT,
            tile_grid_size=settings.CLAHE_TILE_GRID_SIZE
        )

    pil_img = Image.fromarray(face_rgb)
    transform = get_inference_transforms()
    tensor = transform(pil_img)
    return tensor
