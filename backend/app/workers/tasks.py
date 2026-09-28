from pathlib import Path
import asyncio
import logging
from app.database.mongodb import update_job_status, update_job_result, update_job_failed
from app.storage import get_frames_directory_for_job
from app.inference.image_inference import run_image_inference
from app.inference.video_inference import run_video_inference

logger = logging.getLogger("uvicorn.error")


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
      1. Sets status -> 'processing'
      2. Dispatches image or video pipeline (in a worker thread so the event loop stays responsive)
      3. On success: sets status -> 'completed' with metrics
      4. On error: sets status -> 'failed' with friendly error description
    """
    try:
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
        await update_job_failed(job_id, err_msg)

    except ValueError as e:
        err_msg = str(e)
        logger.warning(f"Job {job_id} validation/processing error: {err_msg}")
        await update_job_failed(job_id, err_msg)

    except Exception as e:
        err_msg = f"Unexpected processing error: {str(e)}"
        logger.exception(f"Unhandled exception during job {job_id}: {e}")
        await update_job_failed(job_id, err_msg)
