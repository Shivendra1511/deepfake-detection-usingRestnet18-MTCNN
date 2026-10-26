import threading
from typing import Optional, Tuple, List
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN

from app.config import settings
from app.model.resnet18_detector import get_device

_CACHED_MTCNN: Optional[MTCNN] = None
_MTCNN_LOCK = threading.Lock()

def get_mtcnn(device: Optional[torch.device] = None) -> MTCNN:
    """
    Singleton getter for MTCNN detector with project-specified configurations:
      min_face_size = 20
      thresholds = [0.6, 0.7, 0.7]
      keep_all = True
    """
    global _CACHED_MTCNN
    if _CACHED_MTCNN is not None:
        return _CACHED_MTCNN

    with _MTCNN_LOCK:
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
    mtcnn: Optional[MTCNN] = None,
    max_side: Optional[int] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect faces in an RGB numpy image using MTCNN.
    Large images are downscaled so their longest side is at most max_side
    (default: settings.DETECTION_MAX_SIDE) before detection, and the boxes are
    scaled back to the original resolution. MTCNN cost grows with pixel count,
    so this makes 12MP photos / 4K frames several times faster to process.
    Returns:
      boxes: np.ndarray of shape (N, 4) with bounding boxes [x1, y1, x2, y2] in original-image pixels
      probs: np.ndarray of shape (N,) with detection confidence scores
    If no face is detected, returns (None, None).
    """
    if mtcnn is None:
        mtcnn = get_mtcnn()
    if max_side is None:
        max_side = settings.DETECTION_MAX_SIDE

    img_h, img_w = image_rgb.shape[:2]
    scale = 1.0
    detect_img = image_rgb
    if max_side and max(img_h, img_w) > max_side:
        scale = max_side / float(max(img_h, img_w))
        detect_img = cv2.resize(
            image_rgb,
            (max(1, int(round(img_w * scale))), max(1, int(round(img_h * scale)))),
            interpolation=cv2.INTER_AREA
        )

    # Convert numpy array to PIL Image for facenet-pytorch compatibility
    pil_img = Image.fromarray(detect_img)
    boxes, probs = mtcnn.detect(pil_img)

    if boxes is None or len(boxes) == 0:
        return None, None

    if scale != 1.0:
        boxes = boxes / scale

    return boxes, probs
