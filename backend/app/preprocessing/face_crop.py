import math
from typing import Optional, Tuple, List, Union
import numpy as np
import cv2

from app.config import settings

def select_largest_face(
    boxes: Optional[np.ndarray],
    probs: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """
    Select the largest face bounding box by area: (x2 - x1) * (y2 - y1).
    Returns bounding box array [x1, y1, x2, y2] or None if no faces found.
    """
    if boxes is None or len(boxes) == 0:
        return None

    largest_box = None
    max_area = -1.0

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        if area > max_area:
            max_area = area
            largest_box = box

    return largest_box

def apply_face_padding(
    box: Union[np.ndarray, List[float]],
    img_width: int,
    img_height: int,
    padding: float = 0.2
) -> Tuple[int, int, int, int]:
    """
    Expand bounding box with 20% (padding=0.2) margin as specified in the report:
      w = x2 - x1
      h = y2 - y1
      pad_x = floor(w * 0.2)
      pad_y = floor(h * 0.2)
      x1 = x1 - pad_x
      y1 = y1 - pad_y
      x2 = x2 + pad_x
      y2 = y2 + pad_y
    Clamps coordinates strictly to image boundaries [0, img_width] and [0, img_height].
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    pad_x = int(math.floor(w * padding))
    pad_y = int(math.floor(h * padding))

    new_x1 = max(0, int(round(x1 - pad_x)))
    new_y1 = max(0, int(round(y1 - pad_y)))
    new_x2 = min(img_width, int(round(x2 + pad_x)))
    new_y2 = min(img_height, int(round(y2 + pad_y)))

    return new_x1, new_y1, new_x2, new_y2

def crop_and_resize_face(
    image_rgb: np.ndarray,
    box: Tuple[int, int, int, int],
    target_size: Tuple[int, int] = (224, 224)
) -> Optional[np.ndarray]:
    """
    Crop face from RGB image given padded coordinates and resize to target_size (224, 224).
    Returns resized RGB numpy array or None if crop dimensions are degenerate.
    """
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None

    face_crop = image_rgb[y1:y2, x1:x2]
    if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
        return None

    resized = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_LINEAR)
    return resized

def apply_clahe(
    image_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8
) -> np.ndarray:
    """
    Optional lighting enhancement using CLAHE on the Lightness (L) channel in LAB color space.
      clipLimit = 2.0
      tileGridSize = (8, 8)
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size)
    )
    enhanced_l = clahe.apply(l)

    enhanced_lab = cv2.merge([enhanced_l, a, b])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb
