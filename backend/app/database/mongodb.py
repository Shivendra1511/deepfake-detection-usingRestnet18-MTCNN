from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from app.config import settings

logger = logging.getLogger("uvicorn.error")

_client: Optional[AsyncIOMotorClient] = None

def get_client() -> AsyncIOMotorClient:
    """Singleton getter for Motor AsyncIOMotorClient."""
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(
            settings.MONGO_URL,
            serverSelectionTimeoutMS=3000
        )
    return _client

def get_database() -> AsyncIOMotorDatabase:
    """Retrieve configured database instance."""
    client = get_client()
    return client[settings.MONGO_DB]

def get_jobs_collection() -> AsyncIOMotorCollection:
    """Retrieve jobs collection instance."""
    db = get_database()
    return db[settings.MONGO_JOBS_COLLECTION]

def utc_now_iso() -> str:
    """Timezone-aware UTC timestamp (ends with +00:00) so browsers convert it to local time correctly."""
    return datetime.now(timezone.utc).isoformat()

async def ensure_indexes() -> None:
    """Create a unique index on job_id so status polling does not scan the whole collection."""
    collection = get_jobs_collection()
    await collection.create_index("job_id", unique=True)

async def check_db_connection() -> bool:
    """
    Check if local MongoDB server is responding.
    Returns True if connected, False otherwise.
    """
    try:
        client = get_client()
        await client.admin.command('ping')
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
        logger.warning(f"MongoDB connection check failed: {e}")
        return False

async def create_job(job_id: str, filename: str, file_type: str) -> Dict[str, Any]:
    """
    Insert a newly submitted job into MongoDB with 'queued' status.
    """
    collection = get_jobs_collection()
    now_iso = utc_now_iso()
    doc = {
        "job_id": job_id,
        "filename": filename,
        "type": file_type,
        "status": "queued",
        "score": None,
        "label": None,
        "frames": [],
        "frame_scores": [],
        "processed_frame_count": 0,
        "detected_face_count": 0,
        "error_message": None,
        "created_at": now_iso,
        "updated_at": now_iso
    }
    await collection.insert_one(doc)
    return doc

async def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve job document by job_id. Returns None if not found.
    Excludes MongoDB internal '_id' from returned dict.
    """
    collection = get_jobs_collection()
    doc = await collection.find_one({"job_id": job_id}, {"_id": 0})
    return doc

async def update_job_status(job_id: str, status: str) -> bool:
    """Update job lifecycle status ('queued', 'processing', 'completed', 'failed')."""
    collection = get_jobs_collection()
    res = await collection.update_one(
        {"job_id": job_id},
        {"$set": {"status": status, "updated_at": utc_now_iso()}}
    )
    return res.modified_count > 0

async def update_job_result(
    job_id: str,
    score: float,
    label: str,
    frames: List[Dict[str, Any]] = None,
    frame_scores: List[float] = None,
    processed_frame_count: int = 0,
    detected_face_count: int = 0
) -> bool:
    """
    Update job with final inference results and mark completed.
    """
    collection = get_jobs_collection()
    now_iso = utc_now_iso()
    update_data = {
        "status": "completed",
        "score": round(score, 4),
        "label": label,
        "frames": frames or [],
        "frame_scores": [round(s, 4) for s in (frame_scores or [])],
        "processed_frame_count": processed_frame_count,
        "detected_face_count": detected_face_count,
        "error_message": None,
        "updated_at": now_iso
    }
    res = await collection.update_one(
        {"job_id": job_id},
        {"$set": update_data}
    )
    return res.modified_count > 0

async def update_job_failed(job_id: str, error_message: str) -> bool:
    """
    Mark job as failed and store readable error message.
    """
    collection = get_jobs_collection()
    res = await collection.update_one(
        {"job_id": job_id},
        {
            "$set": {
                "status": "failed",
                "error_message": error_message,
                "updated_at": utc_now_iso()
            }
        }
    )
    return res.modified_count > 0
