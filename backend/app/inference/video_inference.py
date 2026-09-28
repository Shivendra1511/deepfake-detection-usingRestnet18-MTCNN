from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import numpy as np

from app.config import settings
from app.model.resnet18_detector import load_detector, predict_tensor, get_device, classify_probability
from app.preprocessing.mtcnn_detector import get_mtcnn, detect_faces
from app.preprocessing.face_crop import (
    select_largest_face,
    apply_face_padding,
    crop_and_resize_face
)
from app.preprocessing.transforms import preprocess_face_tensor
from app.video.frame_extraction import extract_video_frames

def run_video_inference(
    video_path: Union[str, Path],
    job_id: str,
    output_frames_dir: Path,
    use_clahe: Optional[bool] = None,
    max_frames: Optional[int] = None
) -> Dict[str, Any]:
    """
    Execute full video deepfake detection pipeline:
      1. Extract up to max_frames (default 64) evenly
      2. For each extracted frame:
         - Detect face with MTCNN
         - If face present: pad, crop, resize to 224x224, normalize, predict
         - Store frame probability
      3. Compute mean probability across all valid face frames
      4. Classify video into REAL / UNCERTAIN / FAKE
    """
    max_f = max_frames or settings.MAX_VIDEO_FRAMES
    device = get_device()
    mtcnn = get_mtcnn(device=device)

    # 1. Load model first so missing model fails early before frame extraction
    detector = load_detector()

    # 2. Extract representative frames from video
    extracted_frames = extract_video_frames(
        video_path=video_path,
        output_dir=output_frames_dir,
        job_id=job_id,
        max_frames=max_f
    )

    frame_results: List[Dict[str, Any]] = []
    frame_scores: List[float] = []

    # 3. Process each extracted frame
    for item in extracted_frames:
        frame_rgb = item["frame_rgb"]
        img_h, img_w, _ = frame_rgb.shape

        boxes, probs = detect_faces(frame_rgb, mtcnn=mtcnn)
        has_face = False
        frame_score = None
        frame_label = None

        if boxes is not None and len(boxes) > 0:
            largest_box = select_largest_face(boxes, probs)
            if largest_box is not None:
                padded_box = apply_face_padding(
                    largest_box,
                    img_width=img_w,
                    img_height=img_h,
                    padding=settings.FACE_PADDING
                )
                face_crop = crop_and_resize_face(
                    frame_rgb,
                    padded_box,
                    target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE)
                )

                if face_crop is not None:
                    face_tensor = preprocess_face_tensor(face_crop, use_clahe=use_clahe)
                    score, label = predict_tensor(detector, face_tensor, device=device)
                    has_face = True
                    frame_score = score
                    frame_label = label
                    frame_scores.append(score)

        frame_results.append({
            "frame_index": item["frame_index"],
            "timestamp_sec": item["timestamp_sec"],
            "frame_url": item["frame_url"],
            "face_detected": has_face,
            "score": frame_score,
            "label": frame_label
        })

    # 4. Handle case where no faces were detected in any frame
    if len(frame_scores) == 0:
        raise ValueError(
            f"No faces detected in any of the {len(extracted_frames)} sampled video frames. "
            "Please ensure the video contains clear and visible human faces."
        )

    # 5. Calculate video-level mean probability
    mean_probability = round(float(np.mean(frame_scores)), 4)
    video_label = classify_probability(mean_probability)

    return {
        "score": mean_probability,
        "label": video_label,
        "frames": frame_results,
        "frame_scores": frame_scores,
        "processed_frame_count": len(extracted_frames),
        "detected_face_count": len(frame_scores)
    }
