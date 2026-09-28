from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from app.storage import generate_job_id, save_upload_file, validate_and_classify_file
from app.database.mongodb import create_job
from app.workers.tasks import process_file_async

router = APIRouter(tags=["Upload"])

class UploadResponse(BaseModel):
    job_id: str = Field(..., description="Unique job tracking identifier")
    filename: str = Field(..., description="Name of uploaded file")
    type: str = Field(..., description="Detected media type ('image' or 'video')")
    status: str = Field(default="queued", description="Initial lifecycle status")

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload image or video for deepfake analysis",
    description="Validates media file, generates a unique Job ID, stores file, inserts queued job in MongoDB, and triggers background processing."
)
async def upload_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided or filename is empty.")

    is_valid, media_type = validate_and_classify_file(file.filename)
    if not is_valid or media_type is None:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Supported images: .jpg, .jpeg, .png. Supported videos: .mp4, .mov, .avi"
        )

    # 1. Generate unique Job ID
    job_id = generate_job_id()

    # 2. Save original file securely to uploads/
    saved_path, detected_type = await save_upload_file(job_id, file)

    # 3. Create initial queued document in MongoDB
    try:
        await create_job(job_id=job_id, filename=file.filename, file_type=detected_type)
    except Exception as e:
        # Clean up file if database insert failed
        if saved_path.exists():
            saved_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record job in database: {str(e)}. Please check if MongoDB is running."
        )

    # 4. Schedule asynchronous background processing task
    background_tasks.add_task(process_file_async, job_id, saved_path, detected_type)

    return UploadResponse(
        job_id=job_id,
        filename=file.filename,
        type=detected_type,
        status="queued"
    )
