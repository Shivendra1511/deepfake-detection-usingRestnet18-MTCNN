from pathlib import Path
from typing import List, Dict, Any, Union
import cv2
import numpy as np

def open_video(video_path: Union[str, Path]) -> cv2.VideoCapture:
    """
    Open video file using OpenCV.
    Raises ValueError if video cannot be opened.
    """
    resolved_path = str(Path(video_path).resolve())
    cap = cv2.VideoCapture(resolved_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file at: {resolved_path}. File may be corrupted or unreadable.")
    return cap

def get_video_metadata(cap: cv2.VideoCapture) -> Dict[str, Any]:
    """
    Retrieve metadata from an open VideoCapture stream.
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    return {
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": round(duration_sec, 2)
    }

def get_sample_frame_indices(total_frames: int, max_frames: int = 64) -> List[int]:
    """
    Generate evenly spaced frame indices across total_frames, bounded by max_frames.
    """
    if total_frames <= 0:
        return []
    if total_frames <= max_frames:
        return list(range(total_frames))

    indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    # Deduplicate while preserving order
    return sorted(list(dict.fromkeys(indices)))

def extract_video_frames(
    video_path: Union[str, Path],
    output_dir: Path,
    job_id: str,
    max_frames: int = 64
) -> List[Dict[str, Any]]:
    """
    Extract up to max_frames evenly from the video and save them into output_dir.
    Returns metadata list containing:
      frame_index, timestamp_sec, frame_path, frame_filename, frame_url, frame_rgb
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = open_video(video_path)
    meta = get_video_metadata(cap)

    total_frames = meta["total_frames"]
    fps = meta["fps"]

    if total_frames <= 0:
        cap.release()
        raise ValueError("Video file contains 0 readable frames or is corrupted.")

    sample_indices = set(get_sample_frame_indices(total_frames, max_frames=max_frames))
    extracted: List[Dict[str, Any]] = []

    current_idx = 0
    saved_count = 0

    try:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if current_idx in sample_indices:
                saved_count += 1
                frame_filename = f"frame_{saved_count:03d}.jpg"
                frame_path = output_dir / frame_filename

                # Save frame image on disk as JPEG
                cv2.imwrite(str(frame_path), frame_bgr)

                # Convert BGR -> RGB for MTCNN / PyTorch
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                timestamp_sec = round(current_idx / fps, 2) if fps > 0 else 0.0

                extracted.append({
                    "frame_index": saved_count,
                    "original_frame_number": current_idx,
                    "timestamp_sec": timestamp_sec,
                    "frame_path": frame_path,
                    "frame_filename": frame_filename,
                    "frame_url": f"/frames/{job_id}/{frame_filename}",
                    "frame_rgb": frame_rgb
                })

            current_idx += 1
            if saved_count >= len(sample_indices):
                break
    finally:
        cap.release()

    if not extracted:
        raise ValueError("Could not extract any valid frames from the video file.")

    return extracted
