from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.database.mongodb import get_job

router = APIRouter(prefix="/jobs", tags=["Jobs"])

class JobStatusResponse(BaseModel):
    job_id: str
    filename: str
    type: str
    status: str
    score: Optional[float] = None
    label: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str

class JobResultResponse(BaseModel):
    job_id: str
    filename: str
    type: str
    status: str
    score: Optional[float] = None
    label: Optional[str] = None
    frames: List[Dict[str, Any]] = []
    frame_scores: List[float] = []
    processed_frame_count: int = 0
    detected_face_count: int = 0
    error_message: Optional[str] = None
    created_at: str
    updated_at: str

@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status for polling",
    description="Returns current lifecycle state ('queued', 'processing', 'completed', 'failed') for polling."
)
async def get_job_status(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found."
        )
    return JobStatusResponse(**job)

@router.get(
    "/results/{job_id}",
    response_model=JobResultResponse,
    summary="Get full job inference results",
    description="Returns completed detection results including score, REAL/UNCERTAIN/FAKE label, and extracted video frames."
)
async def get_job_results(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found."
        )
    return JobResultResponse(**job)
