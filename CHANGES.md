# Patch notes: bug fixes and performance

These changes were made after a code review. They were **not run or tested** on the
machine where they were written (it had no Python or Node), so run the checks at the
bottom before you rely on them.

## Training

- **Face crops are now made once and cached.** `training/dataset.py`, `kaggle_train.py`
  and `kaggle_train.ipynb` run MTCNN once per image and save the crop as a PNG
  (`dataset/_face_cache/` locally, `/kaggle/working/face_cache/` on Kaggle).
  Before this, MTCNN ran again for every image in every epoch.
- **Kaggle bug fixed.** With `NUM_WORKERS = 2`, MTCNN ran on CUDA inside forked
  DataLoader workers. That fails, and the old `except Exception` hid the error, so
  the model was trained on full uncropped images. Now the workers only read cached
  crops, so they never touch CUDA.
- **Images with no face are skipped** (and counted in the log). Before, the full
  image was used. Backend inference already rejects images with no face, so training
  and inference now match.
- Big images are downscaled before MTCNN *detection* only. The crop is still taken
  from the full-resolution image.
- `train.py` has a new `--cache_dir` option. If you change the padding or the MTCNN
  settings, delete the cache folder so the crops are made again.
- The evaluation confusion matrix uses `labels=[0, 1]`, so it stays 2x2 even when
  only one class is present.

## Backend

| Change | File |
|---|---|
| `/inference/predict` runs inference in a worker thread instead of blocking the event loop | `api/routes/inference.py` |
| At most `MAX_CONCURRENT_JOBS` (default 1) jobs run at once; the others wait as `queued` | `workers/tasks.py` |
| The uploaded file is deleted after processing (`DELETE_UPLOAD_AFTER_PROCESSING`); frames and results are kept | `workers/tasks.py` |
| A failure to mark a job as failed (for example, MongoDB down) is logged instead of crashing the task | `workers/tasks.py` |
| Upload size limit `MAX_UPLOAD_MB` (default 200), returns HTTP 413 | `storage.py` |
| Timestamps are UTC with a timezone (`+00:00`), so the browser shows the correct local time | `database/mongodb.py` |
| Unique MongoDB index on `job_id`, created at startup | `database/mongodb.py`, `main.py` |
| Video frames are skipped with `cap.grab()`; only the sampled frames are decoded | `video/frame_extraction.py` |
| Frames are streamed one at a time instead of keeping all 64 full-size frames in RAM | `video/frame_extraction.py`, `inference/video_inference.py` |
| All face crops of a video go through ResNet18 in one batch (`predict_batch`) | `model/resnet18_detector.py`, `inference/video_inference.py` |
| `torch.inference_mode()` is used for prediction | `model/resnet18_detector.py` |
| MTCNN detection runs on a downscaled copy of large images/frames (`DETECTION_MAX_SIDE`, default 1280) | `preprocessing/mtcnn_detector.py` |
| Face resize uses `INTER_AREA` when shrinking, which is closer to the PIL resize used in training | `preprocessing/face_crop.py` |
| Model and MTCNN singletons are created under a lock (thread-safe) | `model/resnet18_detector.py`, `preprocessing/mtcnn_detector.py` |
| A 320px `thumb_XXX.jpg` is saved for each frame; `thumbnail_url` is added to the frame results | `video/frame_extraction.py` |

New settings in `config.py` (these can also be set in `backend/.env`):
`DETECTION_MAX_SIDE`, `THUMBNAIL_WIDTH`, `MAX_UPLOAD_MB`, `MAX_CONCURRENT_JOBS`,
`DELETE_UPLOAD_AFTER_PROCESSING`.

## Frontend

- Job polling retries temporary errors (up to 5 times, with backoff) instead of stopping
  on the first error, and it no longer restarts when the parent re-renders.
- Old jobs saved without a timezone are read as UTC (`parseServerDate`).
- The frame grid loads the small thumbnails. Clicking a frame still shows the full frame.
- Uploads have no 30s timeout, so large videos can finish uploading.

## Not changed (worth doing next)

- Keep a separate **test set**. Right now the best epoch is picked on the val set and
  the final metrics are reported on the same set, so they look better than they are.
  Also make sure frames from one video are not in both train and val.
- The 0.40/0.60 thresholds are still hard-coded in the frontend text.

## Checks to run

```bash
cd backend
pip install -r requirements.txt
python -m compileall app
uvicorn app.main:app --reload
```

```bash
cd frontend
npm install
npm run build
```

Then upload one image and one video, and check that the scores look similar to
before (the INTER_AREA resize can change scores a little). Because the training data
changed (cropped faces, no-face images skipped), **retrain the model** to get the
benefit of the training fixes.
