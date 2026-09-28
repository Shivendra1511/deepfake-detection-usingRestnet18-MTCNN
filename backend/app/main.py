from contextlib import asynccontextmanager
from pathlib import Path
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.api.routes import upload, jobs, inference
from app.database.mongodb import check_db_connection
from app.storage import ensure_directories

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure all local storage folders exist
    ensure_directories()
    logger.info("Local storage directories verified (uploads, frames, results, data).")

    # Check MongoDB status on startup
    db_connected = await check_db_connection()
    if db_connected:
        logger.info(f"Successfully connected to MongoDB at {settings.MONGO_URL} (db: {settings.MONGO_DB}).")
    else:
        logger.warning(
            f"Could not connect to MongoDB at {settings.MONGO_URL}. "
            "Please make sure your local MongoDB service is started."
        )

    # Check if trained detector.pt is present
    model_path = Path(settings.MODEL_PATH).resolve()
    if model_path.exists() and model_path.is_file():
        logger.info(f"Found trained model checkpoint at: {model_path}")
    else:
        logger.info(
            f"Notice: Trained model checkpoint not found at: {model_path}. "
            "Train the model using training/train.py and place detector.pt inside backend/data/."
        )

    yield

    logger.info("Application shutting down.")

app = FastAPI(
    title="Deepfake Detection System (ResNet18-MTCNN)",
    description=(
        "Local student-level Deepfake Detection API using PyTorch (ResNet18 transfer learning), "
        "MTCNN face detection, OpenCV frame sampling, and MongoDB job tracking."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for extracted video frames and uploads
app.mount("/frames", StaticFiles(directory=str(settings.FRAMES_DIR)), name="frames")
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOADS_DIR)), name="uploads")

# Include API Routers
app.include_router(upload.router)
app.include_router(jobs.router)
app.include_router(inference.router)

@app.get("/", tags=["General"])
async def root():
    return {
        "project": "Deepfake Detection in Image and Video Using ResNet18-MTCNN Pipeline",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0"
    }

@app.get("/health", tags=["General"])
async def health_check():
    db_ok = await check_db_connection()
    model_exists = Path(settings.MODEL_PATH).is_file()
    return {
        "status": "healthy" if db_ok else "degraded",
        "mongodb_connected": db_ok,
        "model_file_present": model_exists,
        "model_path": settings.MODEL_PATH
    }
