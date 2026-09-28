from pathlib import Path
from typing import Dict, Any, Union, Optional
import cv2
import numpy as np

from app.config import settings
from app.model.resnet18_detector import load_detector, predict_tensor, get_device
from app.preprocessing.mtcnn_detector import get_mtcnn, detect_faces
from app.preprocessing.face_crop import (
    select_largest_face,
    apply_face_padding,
    crop_and_resize_face
)
from app.preprocessing.transforms import preprocess_face_tensor

def run_image_inference(
    image_path: Union[str, Path],
    use_clahe: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Execute full image deepfake detection pipeline:
      1. Read image with OpenCV
      2. Detect faces with MTCNN
      3. Select largest face
      4. Apply 20% padding
      5. Crop & resize to 224x224
      6. Normalize with ImageNet stats
      7. Run ResNet18 -> Logit -> Sigmoid -> Probability
      8. Determine REAL / UNCERTAIN / FAKE label
    """
    resolved_path = str(Path(image_path).resolve())
    bgr = cv2.imread(resolved_path)
    if bgr is None:
        raise ValueError(f"Could not read image file at: {resolved_path}. File may be corrupted.")

    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = img_rgb.shape

    # 1. Detect faces using MTCNN
    device = get_device()
    mtcnn = get_mtcnn(device=device)
    boxes, probs = detect_faces(img_rgb, mtcnn=mtcnn)

    if boxes is None or len(boxes) == 0:
        raise ValueError("No face detected in the uploaded image. Deepfake detection requires a clear facial region.")

    # 2. Select largest face
    largest_box = select_largest_face(boxes, probs)
    if largest_box is None:
        raise ValueError("No valid face bounding box found in image.")

    # 3. Apply 20% padding
    padded_box = apply_face_padding(
        largest_box,
        img_width=img_w,
        img_height=img_h,
        padding=settings.FACE_PADDING
    )

    # 4. Crop and resize to 224x224
    face_crop = crop_and_resize_face(
        img_rgb,
        padded_box,
        target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE)
    )
    if face_crop is None:
        raise ValueError("Failed to crop facial region from image.")

    # 5. Preprocess into ImageNet normalized tensor
    face_tensor = preprocess_face_tensor(face_crop, use_clahe=use_clahe)

    # 6. Load fine-tuned ResNet18 detector (raises clear error if detector.pt is missing)
    detector = load_detector()

    # 7. Run inference
    score, label = predict_tensor(detector, face_tensor, device=device)

    return {
        "score": score,
        "label": label,
        "face_detected": True,
        "bounding_box": [int(coord) for coord in padded_box],
        "image_size": [img_w, img_h]
    }
