from pathlib import Path
from typing import List, Dict, Any, Union, Iterator
import cv2
import numpy as np

from app.config import settings

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
    return sorted(int(i) for i in dict.fromkeys(indices))

def _save_thumbnail(frame_bgr: np.ndarray, thumb_path: Path, width: int) -> bool:
    """Save a small JPEG preview for the frontend gallery. Returns False if nothing was written."""
    h, w = frame_bgr.shape[:2]
    if w <= width:
        return False
    new_h = max(1, int(round(h * width / float(w))))
    thumb = cv2.resize(frame_bgr, (width, new_h), interpolation=cv2.INTER_AREA)
    return bool(cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80]))

def iter_video_frames(
    video_path: Union[str, Path],
    output_dir: Path,
    job_id: str,
    max_frames: int = 64
) -> Iterator[Dict[str, Any]]:
    """
    Yield up to max_frames evenly spaced frames from the video, one at a time.

    Performance notes:
      - Non-sampled frames are skipped with cap.grab() (no colour conversion / copy),
        and only sampled frames are decoded with cap.retrieve().
      - Frames are yielded one by one instead of keeping all full-resolution frames
        in RAM (64 x 4K frames would be ~1.6 GB).

    Each item contains:
      frame_index, original_frame_number, timestamp_sec, frame_path, frame_filename,
      frame_url, thumbnail_url, frame_rgb
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
    last_index = max(sample_indices)

    current_idx = 0
    saved_count = 0

    try:
        while current_idx <= last_index:
            if not cap.grab():
                break

            if current_idx in sample_indices:
                ret, frame_bgr = cap.retrieve()
                if ret and frame_bgr is not None:
                    saved_count += 1
                    frame_filename = f"frame_{saved_count:03d}.jpg"
                    frame_path = output_dir / frame_filename

                    # Save full frame (for the preview modal) and a small thumbnail (for the grid)
                    cv2.imwrite(str(frame_path), frame_bgr)
                    thumb_filename = f"thumb_{saved_count:03d}.jpg"
                    if _save_thumbnail(frame_bgr, output_dir / thumb_filename, settings.THUMBNAIL_WIDTH):
                        thumbnail_url = f"/frames/{job_id}/{thumb_filename}"
                    else:
                        thumbnail_url = f"/frames/{job_id}/{frame_filename}"

                    # Convert BGR -> RGB for MTCNN / PyTorch
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    timestamp_sec = round(current_idx / fps, 2) if fps > 0 else 0.0

                    yield {
                        "frame_index": saved_count,
                        "original_frame_number": current_idx,
                        "timestamp_sec": timestamp_sec,
                        "frame_path": frame_path,
                        "frame_filename": frame_filename,
                        "frame_url": f"/frames/{job_id}/{frame_filename}",
                        "thumbnail_url": thumbnail_url,
                        "frame_rgb": frame_rgb
                    }

            current_idx += 1
    finally:
        cap.release()

    if saved_count == 0:
        raise ValueError("Could not extract any valid frames from the video file.")

def extract_video_frames(
    video_path: Union[str, Path],
    output_dir: Path,
    job_id: str,
    max_frames: int = 64
) -> List[Dict[str, Any]]:
    """
    List version of iter_video_frames (keeps all frames in memory).
    Prefer iter_video_frames for long / high-resolution videos.
    """
    return list(iter_video_frames(video_path, output_dir, job_id, max_frames=max_frames))
