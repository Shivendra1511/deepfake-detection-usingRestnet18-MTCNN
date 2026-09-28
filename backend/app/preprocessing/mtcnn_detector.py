from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN

from app.config import settings
from app.model.resnet18_detector import get_device

_CACHED_MTCNN: Optional[MTCNN] = None

def get_mtcnn(device: Optional[torch.device] = None) -> MTCNN:
    """
    Singleton getter for MTCNN detector with project-specified configurations:
      min_face_size = 20
      thresholds = [0.6, 0.7, 0.7]
      keep_all = True
    """
    global _CACHED_MTCNN
    if _CACHED_MTCNN is None:
        if device is None:
            device = get_device()
        _CACHED_MTCNN = MTCNN(
            image_size=settings.IMAGE_SIZE,
            margin=0,
            min_face_size=settings.MIN_FACE_SIZE,
            thresholds=settings.MTCNN_THRESHOLDS,
            keep_all=True,
            post_process=False,
            device=device
        )
    return _CACHED_MTCNN

def detect_faces(
    image_rgb: np.ndarray,
    mtcnn: Optional[MTCNN] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect faces in an RGB numpy image using MTCNN.
    Returns:
      boxes: np.ndarray of shape (N, 4) with bounding boxes [x1, y1, x2, y2]
      probs: np.ndarray of shape (N,) with detection confidence scores
    If no face is detected, returns (None, None).
    """
    if mtcnn is None:
        mtcnn = get_mtcnn()

    # Convert numpy array to PIL Image for facenet-pytorch compatibility
    pil_img = Image.fromarray(image_rgb)
    boxes, probs = mtcnn.detect(pil_img)

    if boxes is None or len(boxes) == 0:
        return None, None

    return boxes, probs
