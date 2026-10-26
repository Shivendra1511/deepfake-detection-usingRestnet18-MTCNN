from pathlib import Path
from typing import Any, Callable, Optional
import asyncio
import logging
from app.config import settings
from app.database.mongodb import update_job_status, update_job_result, update_job_failed
from app.storage import get_frames_directory_for_job, delete_file_quietly
from app.inference.image_inference import run_image_inference
from app.inference.video_inference import run_video_inference

logger = logging.getLogger("uvicorn.error")

# Limits how many inference pipelines run at the same time. Several videos processed in
# parallel would otherwise compete for CPU/GPU and RAM and all become slow (or crash).
# Created lazily so it binds to the running event loop.
_job_slots: Optional[asyncio.Semaphore] = None


def _get_job_slots() -> asyncio.Semaphore:
    global _job_slots
    if _job_slots is None:
        _job_slots = asyncio.Semaphore(max(1, settings.MAX_CONCURRENT_JOBS))
    return _job_slots


async def run_in_job_slot(func: Callable[..., Any], *args: Any) -> Any:
    """Run a blocking inference function in a worker thread, waiting for a free job slot."""
    async with _get_job_slots():
        return await asyncio.to_thread(func, *args)


def _run_image_job(file_path: Path) -> dict:
    return run_image_inference(image_path=file_path)


def _run_video_job(file_path: Path, job_id: str, frames_dir: Path) -> dict:
    return run_video_inference(
        video_path=file_path,
        job_id=job_id,
        output_frames_dir=frames_dir,
    )


async def process_file_async(job_id: str, file_path: Path, file_type: str) -> None:
    """
    FastAPI BackgroundTask runner:
      1. Waits for a free job slot (status stays 'queued' meanwhile)
      2. Sets status -> 'processing'
      3. Dispatches image or video pipeline (in a worker thread so the event loop stays responsive)
      4. On success: sets status -> 'completed' with metrics
      5. On error: sets status -> 'failed' with friendly error description
      6. Deletes the uploaded file (results and frames are kept)
    """
    try:
        async with _get_job_slots():
            logger.info(f"Starting background processing for job {job_id} ({file_type})")
            await update_job_status(job_id, "processing")

            if file_type == "image":
                result = await asyncio.to_thread(_run_image_job, file_path)
                await update_job_result(
                    job_id=job_id,
                    score=result["score"],
                    label=result["label"],
                    frames=[],
                    frame_scores=[result["score"]],
                    processed_frame_count=1,
                    detected_face_count=1,
                )
            elif file_type == "video":
                frames_dir = get_frames_directory_for_job(job_id)
                result = await asyncio.to_thread(_run_video_job, file_path, job_id, frames_dir)
                await update_job_result(
                    job_id=job_id,
                    score=result["score"],
                    label=result["label"],
                    frames=result["frames"],
                    frame_scores=result["frame_scores"],
                    processed_frame_count=result["processed_frame_count"],
                    detected_face_count=result["detected_face_count"],
                )
            else:
                raise ValueError(f"Unknown media type: {file_type}")

        logger.info(f"Job {job_id} successfully processed and marked completed.")

    except FileNotFoundError as e:
        err_msg = str(e)
        logger.error(f"Job {job_id} failed due to missing model: {err_msg}")
        await _mark_failed(job_id, err_msg)

    except ValueError as e:
        err_msg = str(e)
        logger.warning(f"Job {job_id} validation/processing error: {err_msg}")
        await _mark_failed(job_id, err_msg)

    except Exception as e:
        err_msg = f"Unexpected processing error: {str(e)}"
        logger.exception(f"Unhandled exception during job {job_id}: {e}")
        await _mark_failed(job_id, err_msg)

    finally:
        if settings.DELETE_UPLOAD_AFTER_PROCESSING:
            delete_file_quietly(file_path)


async def _mark_failed(job_id: str, err_msg: str) -> None:
    """Store the failure; if MongoDB itself is down, log instead of raising from the background task."""
    try:
        await update_job_failed(job_id, err_msg)
    except Exception as db_err:
        logger.error(f"Could not mark job {job_id} as failed in MongoDB: {db_err}")
