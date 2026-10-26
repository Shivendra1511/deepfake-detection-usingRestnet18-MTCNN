from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field

from app.storage import generate_job_id, save_upload_file, validate_and_classify_file
from app.inference.image_inference import run_image_inference
from app.workers.tasks import run_in_job_slot

router = APIRouter(prefix="/inference", tags=["Inference"])

class DirectPredictResponse(BaseModel):
    filename: str
    type: str = "image"
    score: float = Field(..., description="Fake probability score between 0.0 and 1.0")
    label: str = Field(..., description="Classification result: REAL, UNCERTAIN, or FAKE")
    face_detected: bool = True
    bounding_box: Optional[List[int]] = None

@router.post(
    "/predict",
    response_model=DirectPredictResponse,
    summary="Direct synchronous image inference",
    description="Allows direct testing of an image file without background queue. Returns immediate prediction score and label."
)
async def direct_image_predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    is_valid, media_type = validate_and_classify_file(file.filename)
    if not is_valid or media_type != "image":
        raise HTTPException(
            status_code=400,
            detail="Direct prediction endpoint currently supports single images (.jpg, .jpeg, .png). For videos, please use /upload."
        )

    temp_id = f"test_{generate_job_id()}"
    saved_path, _ = await save_upload_file(temp_id, file)

    try:
        # Run the blocking MTCNN + ResNet pipeline in a worker thread so the event loop
        # (and every other request, e.g. job polling) is not frozen while it runs.
        result = await run_in_job_slot(run_image_inference, saved_path)
        return DirectPredictResponse(
            filename=file.filename,
            type="image",
            score=result["score"],
            label=result["label"],
            face_detected=result["face_detected"],
            bounding_box=result.get("bounding_box")
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )
    finally:
        # Cleanup temporary uploaded file
        if saved_path.exists():
            saved_path.unlink()
