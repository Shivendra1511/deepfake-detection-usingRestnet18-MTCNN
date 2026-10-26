import uuid
from pathlib import Path
from typing import Tuple, Optional
from fastapi import UploadFile, HTTPException

from app.config import settings

# Supported file extensions
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}

# Ensure required directories exist
def ensure_directories() -> None:
    settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    settings.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

# Run directory creation on import
ensure_directories()

def generate_job_id() -> str:
    """Generate a unique hex job ID."""
    return uuid.uuid4().hex

def validate_and_classify_file(filename: str) -> Tuple[bool, Optional[str]]:
    """
    Validate filename extension and determine if it is an image or video.
    Returns: (is_valid, 'image' | 'video' | None)
    """
    if not filename:
        return False, None

    ext = Path(filename).suffix.lower()
    if ext in ALLOWED_IMAGE_EXTENSIONS:
        return True, "image"
    elif ext in ALLOWED_VIDEO_EXTENSIONS:
        return True, "video"
    return False, None

def get_safe_filename(job_id: str, original_filename: str) -> str:
    """
    Sanitize and prefix filename with job ID to prevent collisions and path traversal.
    """
    clean_name = Path(original_filename).name
    ext = Path(clean_name).suffix.lower()
    return f"{job_id}{ext}"

async def save_upload_file(job_id: str, upload_file: UploadFile) -> Tuple[Path, str]:
    """
    Save incoming UploadFile to the local uploads directory.
    Rejects files larger than settings.MAX_UPLOAD_MB with HTTP 413.
    Returns: (saved_path, file_type) where file_type is 'image' or 'video'.
    """
    original_name = upload_file.filename or "upload"
    is_valid, file_type = validate_and_classify_file(original_name)
    if not is_valid or file_type is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: images ({', '.join(ALLOWED_IMAGE_EXTENSIONS)}) and videos ({', '.join(ALLOWED_VIDEO_EXTENSIONS)})."
        )

    safe_name = get_safe_filename(job_id, original_name)
    dest_path = settings.UPLOADS_DIR / safe_name
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    # Write content to local disk safely
    written = 0
    try:
        with open(dest_path, "wb") as f:
            while chunk := await upload_file.read(1024 * 1024):  # 1MB chunks
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum allowed size is {settings.MAX_UPLOAD_MB} MB."
                    )
                f.write(chunk)
    except HTTPException:
        dest_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    return dest_path, file_type

def delete_file_quietly(path: Path) -> None:
    """Remove a file if it exists, ignoring errors (used for upload cleanup)."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass

def get_frames_directory_for_job(job_id: str) -> Path:
    """Return and create frames directory for a specific job."""
    job_frames_dir = settings.FRAMES_DIR / job_id
    job_frames_dir.mkdir(parents=True, exist_ok=True)
    return job_frames_dir
