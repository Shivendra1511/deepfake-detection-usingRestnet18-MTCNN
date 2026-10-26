from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base backend directory: d:\prpe\deepfake\backend
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # MongoDB Configuration
    MONGO_URL: str = "mongodb://localhost:27017"
    MONGO_DB: str = "deepfake_detection"
    MONGO_JOBS_COLLECTION: str = "jobs"

    # Model & Checkpoint Configuration
    MODEL_PATH: str = str(BASE_DIR / "data" / "detector.pt")
    IMAGE_SIZE: int = 224

    # MTCNN Face Detection Parameters
    MIN_FACE_SIZE: int = 20
    MTCNN_THRESHOLDS: List[float] = [0.6, 0.7, 0.7]
    FACE_PADDING: float = 0.2
    # Images/frames larger than this (longest side, px) are downscaled for MTCNN detection only;
    # the face crop is still taken from the full-resolution image.
    DETECTION_MAX_SIDE: int = 1280

    # Classification Thresholds (REAL < LOWER_THRESHOLD, FAKE > UPPER_THRESHOLD, else UNCERTAIN)
    LOWER_THRESHOLD: float = 0.4
    UPPER_THRESHOLD: float = 0.6

    # Video Processing
    MAX_VIDEO_FRAMES: int = 64
    THUMBNAIL_WIDTH: int = 320

    # Uploads & Job Processing
    MAX_UPLOAD_MB: int = 200
    MAX_CONCURRENT_JOBS: int = 1
    DELETE_UPLOAD_AFTER_PROCESSING: bool = True

    # Optional CLAHE Preprocessing
    USE_CLAHE: bool = False
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: int = 8

    # Server Configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    # Directories
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    FRAMES_DIR: Path = BASE_DIR / "frames"
    RESULTS_DIR: Path = BASE_DIR / "results"
    DATA_DIR: Path = BASE_DIR / "data"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
