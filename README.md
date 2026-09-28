# Deepfake Detection in Image and Video Using ResNet18-MTCNN Pipeline

A complete, runnable, local, student-level Deepfake Detection System built for a **BTech Final-Year Computer Science & Engineering Project**.

This application detects manipulated or deepfake facial content in both images and videos using an **ImageNet-pretrained ResNet18** transfer-learning classifier coupled with a multi-stage **MTCNN (Multi-task Cascaded Convolutional Networks)** face detection and 20% padded cropping pipeline.

---

## 1. System Architecture & Methodology

```text
                                  +-----------------------------+
                                  |   React + Vite Frontend     |
                                  | (TypeScript + Clean Dark UI)|
                                  +--------------+--------------+
                                                 |
                                     HTTP POST /upload (FormData)
                                                 |
                                                 v
                                  +-----------------------------+
                                  |       FastAPI Backend       |
                                  |   (Monolithic Local API)    |
                                  +-------+--------------+------+
                                          |              |
                    Job Document (status: "queued")      | Saves uploaded media
                                          |              |
                                          v              v
                                  +---------------+ +-------------------+
                                  | Local MongoDB | | backend/uploads/  |
                                  | (Port 27017)  | +-------------------+
                                  +---------------+
                                          |
                        BackgroundTasks: process_file_async()
                                          |
                 +------------------------+-------------------------+
                 | (IMAGE PIPELINE)                                 | (VIDEO PIPELINE)
                 v                                                  v
     +-----------------------+                         +---------------------------+
     | Read Image via OpenCV |                         | OpenCV: Sample <=64 frames|
     +-----------+-----------+                         +-------------+-------------+
                 |                                                   |
                 |                                        Save frames to frames/<job_id>/
                 |                                                   |
                 v                                                   v
     +-----------------------+                         +---------------------------+
     | MTCNN Face Detection  |                         | MTCNN Face Detection per  |
     | (min_size=20, [0.6..])|                         | sampled frame             |
     +-----------+-----------+                         +-------------+-------------+
                 |                                                   |
                 v                                                   v
     +-----------------------+                         +---------------------------+
     | Largest Face Selected |                         | Largest Face & 20% Padding|
     | & 20% Padded Crop     |                         | per valid frame           |
     +-----------+-----------+                         +-------------+-------------+
                 |                                                   |
                 v                                                   v
     +-----------------------+                         +---------------------------+
     | Resize to 224 x 224 & |                         | Resize to 224 x 224 &     |
     | ImageNet Normalization|                         | ImageNet Normalization    |
     +-----------+-----------+                         +-------------+-------------+
                 |                                                   |
                 v                                                   v
     +-----------------------+                         +---------------------------+
     | Fine-tuned ResNet18   |                         | ResNet18 Inference per    |
     | Single Logit Output   |                         | detected face             |
     +-----------+-----------+                         +-------------+-------------+
                 |                                                   |
                 v                                                   v
     +-----------------------+                         +---------------------------+
     | Sigmoid -> Probability|                         | Calculate Mean Probability|
     | [0.00 to 1.00]        |                         | P = (p1 + p2 + ... + pN)/N|
     +-----------+-----------+                         +-------------+-------------+
                 \                                                  /
                  \                                                /
                   v                                              v
               +------------------------------------------------------+
               | Three-State Decision Boundary:                       |
               |   Score < 0.40       => REAL                         |
               |   0.40 <= Score <= 0.60 => UNCERTAIN                 |
               |   Score > 0.60       => FAKE                         |
               +--------------------------+---------------------------+
                                          |
                         Update MongoDB: status = "completed"
                                          |
                         Frontend Polling: GET /jobs/{job_id}
                                          |
                                          v
               +------------------------------------------------------+
               | Frontend Results Presentation:                       |
               | - REAL / UNCERTAIN / FAKE Badge                      |
               | - Fake Probability Percentage Meter                  |
               | - Interactive Video Frame Analysis Gallery           |
               +------------------------------------------------------+
```

---

## 2. Project Directory Structure

```text
deepfake/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI application entrypoint, lifespan & static mounts
│   │   ├── config.py                   # Pydantic settings, paths, thresholds, and MTCNN parameters
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes/
│   │   │       ├── __init__.py
│   │   │       ├── upload.py           # POST /upload endpoint
│   │   │       ├── jobs.py             # GET /jobs/{id} and GET /jobs/results/{id}
│   │   │       └── inference.py        # POST /inference/predict (direct image testing)
│   │   │
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   └── mongodb.py              # Motor async MongoDB client, collections & job queries
│   │   │
│   │   ├── model/
│   │   │   ├── __init__.py
│   │   │   └── resnet18_detector.py    # ResNet18 model definition, checkpoint loader, logit classifier
│   │   │
│   │   ├── preprocessing/
│   │   │   ├── __init__.py
│   │   │   ├── mtcnn_detector.py       # facenet-pytorch MTCNN face detection wrapper
│   │   │   ├── face_crop.py            # Largest face selection, 20% padding, crop & optional CLAHE
│   │   │   └── transforms.py           # ImageNet normalization and preprocessing tensors
│   │   │
│   │   ├── inference/
│   │   │   ├── __init__.py
│   │   │   ├── image_inference.py      # End-to-end image deepfake detection pipeline
│   │   │   └── video_inference.py      # End-to-end video deepfake detection & frame aggregation
│   │   │
│   │   ├── video/
│   │   │   ├── __init__.py
│   │   │   └── frame_extraction.py     # OpenCV video capture, sampling <=64 frames, disk saving
│   │   │
│   │   ├── workers/
│   │   │   ├── __init__.py
│   │   │   └── tasks.py                # FastAPI BackgroundTasks runner (queued -> processing -> completed)
│   │   │
│   │   └── storage.py                  # Path sanitization, UUID job generator, folder initialization
│   │
│   ├── data/
│   │   ├── .gitkeep
│   │   └── detector.pt                 # <-- Trained checkpoint location (generated after training)
│   ├── uploads/                        # Temporary local storage for uploaded raw user files
│   ├── frames/                         # Extracted video frames per job (frames/<job_id>/)
│   ├── results/                        # Local results cache
│   ├── .env                            # Backend configuration variables
│   ├── .gitignore
│   └── requirements.txt                # Python backend dependencies
│
├── training/
│   ├── train.py                        # Complete PyTorch training pipeline with ReduceLROnPlateau
│   ├── dataset.py                      # PyTorch Dataset loading real/fake folders with MTCNN face crops
│   ├── model.py                        # Pretrained ResNet18 with Dropout(0.5) + Linear(512, 1) head
│   ├── transforms.py                   # Training augmentations (flip, jitter, rotation) & Val transforms
│   ├── evaluate.py                     # Evaluation script (Accuracy, Precision, Recall, F1, Confusion Matrix)
│   ├── utils.py                        # Reproducibility seeds and training curves plotting
│   └── README.md                       # Dedicated training documentation
│
├── dataset/
│   ├── train/
│   │   ├── real/                       # Real face training images (.jpg, .png, etc.)
│   │   └── fake/                       # Manipulated face training images
│   └── val/
│       ├── real/                       # Real validation images
│       └── fake/                       # Manipulated validation images
│
├── results/
│   └── training/                       # Generated loss and accuracy plots (training_loss.png, etc.)
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Uploader.tsx            # Drag & drop file selector with format validation
│   │   │   ├── JobStatus.tsx           # Asynchronous polling component with step progression
│   │   │   ├── ResultCard.tsx          # REAL / UNCERTAIN / FAKE verdict, probability meter
│   │   │   └── FrameGallery.tsx        # Video frame grid with individual frame scores and modal view
│   │   │
│   │   ├── pages/
│   │   │   ├── Home.tsx                # Hero section, feature pills, upload and active report area
│   │   │   ├── Results.tsx             # Past job lookup by Job ID
│   │   │   └── About.tsx               # Academic technical documentation of the system
│   │   │
│   │   ├── services/
│   │   │   └── api.ts                  # Axios client connecting to FastAPI backend
│   │   │
│   │   ├── App.tsx                     # Main layout, tab navigation, and backend health monitor
│   │   ├── main.tsx                    # React DOM entrypoint
│   │   └── index.css                   # Custom responsive styling and dark mode UI
│   │
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── .env                            # Frontend API base URL (VITE_API_BASE_URL)
│   └── .gitignore
│
├── README.md                           # Master Project Documentation
└── .gitignore
```

---

## 3. Technology Stack

- **Frontend:** React 18, Vite, TypeScript, Axios, Lucide React Icons.
- **Backend:** Python 3.9+, FastAPI, Uvicorn, Pydantic, python-dotenv, Motor (Async MongoDB Driver).
- **Database:** Local MongoDB Community Server (`mongodb://localhost:27017`).
- **Machine Learning & Deep Learning:** PyTorch, Torchvision (`ResNet18`), ImageNet weights.
- **Computer Vision:** OpenCV (`cv2`), Pillow (`PIL`), `facenet-pytorch` (MTCNN).
- **Evaluation & Graphs:** Scikit-Learn (Precision, Recall, F1, Confusion Matrix), Matplotlib.

---

## 4. Prerequisites

1. **Operating System:** Windows 10 or Windows 11.
2. **Python:** Python 3.9 or higher (Python 3.12 recommended). Verify in PowerShell:
   ```powershell
   python --version
   ```
3. **Node.js & npm:** Node.js v18 or higher. Verify in PowerShell:
   ```powershell
   node --version
   npm --version
   ```
4. **MongoDB:** MongoDB Community Server installed and running locally on default port `27017`.

---

## 5. Local Setup Instructions

Open **Windows PowerShell** and follow these step-by-step instructions.

### Step 5.1: Verify MongoDB is Running

If MongoDB is installed as a Windows Service, check its status:

```powershell
Get-Service -Name *mongo*
```

If it is stopped, start it:

```powershell
Start-Service MongoDB
```

*(Optional: You can open MongoDB Compass and connect to `mongodb://localhost:27017` to inspect databases and collections in a visual GUI).*

---

### Step 5.2: Setup Backend Virtual Environment & Install Dependencies

Open a PowerShell terminal in the project root:

```powershell
cd d:\prpe\deepfake\backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install required Python packages
pip install -r requirements.txt
```

---

### Step 5.3: Setup Frontend Dependencies

Open a second PowerShell terminal:

```powershell
cd d:\prpe\deepfake\frontend

# Install Node modules
npm install
```

---

## 6. Training the Model (Transfer Learning)

The system does NOT use dummy or fake weights. To enable deepfake detection, you train the fine-tuned ResNet18 model on your dataset.

### Step 6.1: Organize Your Dataset

Place your images inside `dataset/`:

```text
dataset/
├── train/
│   ├── real/    # Authentic face images (.jpg, .png, .webp)
│   └── fake/    # Manipulated/deepfake face images
└── val/
    ├── real/    # Validation real images
    └── fake/    # Validation fake images
```

### Step 6.2: Run Training

In your PowerShell terminal with the virtual environment activated:

```powershell
cd d:\prpe\deepfake\training
python train.py --epochs 10 --batch_size 16 --lr 0.0001
```

During training, the terminal will print epoch metrics:
- Training Loss
- Validation Loss
- Validation Accuracy (%)
- Current Learning Rate

At the end of training:
1. The best model checkpoint is automatically saved to:
   ```text
   d:\prpe\deepfake\backend\data\detector.pt
   ```
2. Training loss, validation loss, and validation accuracy curves are saved into:
   ```text
   d:\prpe\deepfake\results\training\
   ├── training_loss.png
   ├── validation_loss.png
   └── validation_accuracy.png
   ```

### Step 6.2b: Train on Kaggle (if your laptop is too slow)

Use the self-contained files:

- `training/kaggle_train.py`
- `training/kaggle_train.ipynb`

1. Zip `train/` and `val/` (same folder layout as above) and upload as a Kaggle Dataset.
2. New Kaggle Notebook → GPU → Add that dataset as input.
3. Set `DATASET_DIR = Path("/kaggle/input/YOUR_DATASET_NAME")`.
4. Run the notebook (or `kaggle_train.py`).
5. Download `/kaggle/working/detector.pt` and copy it to `backend/data/detector.pt`.

The checkpoint architecture is the same as local training, so the FastAPI app can load it without code changes.

### Step 6.3: Evaluate the Model (Optional)

Run the standalone evaluation script to calculate Precision, Recall, F1-Score, and Confusion Matrix:

```powershell
python evaluate.py
```

A detailed JSON report will be saved to `results/training/evaluation_report.json`.

---

## 7. Running the Application

### Step 7.1: Start the FastAPI Backend

In your backend PowerShell terminal:

```powershell
cd d:\prpe\deepfake\backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Backend API: `http://127.0.0.1:8000`
- Interactive Swagger Documentation: `http://127.0.0.1:8000/docs`
- Health Endpoint: `http://127.0.0.1:8000/health`

### Step 7.2: Start the React Frontend

In your frontend PowerShell terminal:

```powershell
cd d:\prpe\deepfake\frontend
npm run dev
```

- Open your browser at: `http://localhost:5173`

---

## 8. End-to-End User Workflow

1. Navigate to `http://localhost:5173`.
2. Look at the top navigation bar: The health badge shows **"Model Ready"** (green) once `backend/data/detector.pt` is present.
3. Click or drag an **Image** (`.jpg`, `.png`) or **Video** (`.mp4`, `.mov`, `.avi`) into the drop zone.
4. Click **"Analyze Media"**.
5. The backend validates the format, saves the file in `backend/uploads/`, creates a MongoDB document (`status: "queued"`), and schedules asynchronous inference.
6. The frontend receives the `job_id` and automatically polls `GET /jobs/{job_id}` every 2 seconds.
7. The backend runs the MTCNN face detection, selects the largest face, pads by 20%, resizes to 224×224, and evaluates using ResNet18.
8. When processing completes (`status: "completed"`), the frontend displays:
   - **Verdict Banner:** `REAL` (green), `UNCERTAIN` (amber), or `FAKE` (red).
   - **Manipulation Probability Meter:** Displays exact percentage (e.g. `84.2%`).
   - **Metadata:** File name, media type, timestamp, processed frames count.
   - **Video Frame Analysis Gallery (for videos):** Responsive grid of all sampled frames with individual timestamps, face detection markers, and frame-level probabilities. Clicking any frame opens a full-resolution inspection modal.
9. Click **"Lookup Result"** in the navigation bar to retrieve any past analysis by its Job ID.

---

## 9. Decision Boundaries & Thresholds

The system classifies prediction probability according to configurable boundaries:

$$\text{Probability} = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$$

| Range | Classification Label | Interpretation |
|---|---|---|
| **Probability < 0.40** | **REAL** | Strong evidence of authentic facial features. |
| **0.40 ≤ Probability ≤ 0.60** | **UNCERTAIN** | Borderline zone; inconclusive, manual review recommended. |
| **Probability > 0.60** | **FAKE** | Strong evidence of synthetic deepfake manipulation. |

*Note on Probability vs. Accuracy: The probability score indicates the model's assessed likelihood that a given sample contains deepfake manipulation. It is distinct from historical model test accuracy.*

---

## 10. API Specification & Documentation

FastAPI automatically generates interactive Swagger documentation accessible at `http://127.0.0.1:8000/docs`.

### Key Endpoints

| Method | Path | Request Body / Params | Description |
|---|---|---|---|
| `POST` | `/upload` | `multipart/form-data` (`file`) | Uploads media, generates Job ID, creates queued MongoDB record, launches background task. |
| `GET` | `/jobs/{job_id}` | Path parameter: `job_id` | Returns current job status (`queued`, `processing`, `completed`, `failed`). Used for frontend polling. |
| `GET` | `/jobs/results/{job_id}` | Path parameter: `job_id` | Returns complete inference report (score, label, frame details, error message). |
| `POST` | `/inference/predict` | `multipart/form-data` (`file`) | Direct synchronous image inference endpoint for quick testing without background queuing. |
| `GET` | `/health` | None | Returns backend status, MongoDB connection state, and model presence. |

---

## 11. MongoDB Document Structure

Every submitted job is recorded in the `jobs` collection of the `deepfake_detection` database:

```json
{
  "job_id": "4a7f28bc9c8141f2a3df07e1c8d6729a",
  "filename": "interview_clip.mp4",
  "type": "video",
  "status": "completed",
  "score": 0.8145,
  "label": "FAKE",
  "frames": [
    {
      "frame_index": 1,
      "timestamp_sec": 0.0,
      "frame_url": "/frames/4a7f28bc9c8141f2a3df07e1c8d6729a/frame_001.jpg",
      "face_detected": true,
      "score": 0.8421,
      "label": "FAKE"
    }
  ],
  "frame_scores": [0.8421, 0.7932, 0.8082],
  "processed_frame_count": 64,
  "detected_face_count": 58,
  "error_message": null,
  "created_at": "2026-09-27T10:15:30.123456",
  "updated_at": "2026-09-27T10:15:42.654321"
}
```

---

## 12. Troubleshooting & Common Questions

### 1. "Trained model not found. Please train the model and place detector.pt inside backend/data/."
- **Cause:** The backend was queried for inference before the model was trained.
- **Solution:** Place your dataset in `dataset/train` and `dataset/val`, run `python training/train.py`, and ensure `backend/data/detector.pt` is generated.

### 2. "No face detected in the uploaded image / video."
- **Cause:** MTCNN could not detect a human face in the file with sufficient confidence (`min_face_size=20`, `thresholds=[0.6, 0.7, 0.7]`).
- **Solution:** Ensure the image or video clearly displays the subject's face without extreme obstruction or heavy motion blur.

### 3. "Failed to record job in database: ... Please check if MongoDB is running."
- **Cause:** The local MongoDB service is not started on `localhost:27017`.
- **Solution:** In PowerShell (Run as Administrator), execute `Start-Service MongoDB`. Verify with `Get-Service -Name *mongo*`.

### 4. CORS Errors in Browser Console
- **Cause:** React is running on a port other than `http://localhost:5173`.
- **Solution:** Check `backend/app/config.py`. The `CORS_ORIGINS` list includes `http://localhost:5173`, `http://127.0.0.1:5173`, and `http://localhost:3000`.

---

## 13. End-to-End Verification Checklist

Use this checklist to confirm the entire project is operating properly:

- [ ] Local MongoDB Server is running on `mongodb://localhost:27017`.
- [ ] Python virtual environment created and `requirements.txt` installed.
- [ ] FastAPI backend starts without errors on `http://127.0.0.1:8000`.
- [ ] FastAPI Swagger documentation is accessible at `http://127.0.0.1:8000/docs`.
- [ ] Frontend dependencies installed via `npm install`.
- [ ] React frontend launches on `http://localhost:5173`.
- [ ] Dataset prepared with real and fake samples in `dataset/`.
- [ ] Model training executed with `python training/train.py`.
- [ ] Best model checkpoint saved to `backend/data/detector.pt`.
- [ ] Training and validation loss/accuracy plots saved in `results/training/`.
- [ ] Model evaluation executed with `python training/evaluate.py`.
- [ ] Image upload creates Job ID and records `queued` status in MongoDB.
- [ ] Background inference completes and updates MongoDB to `completed`.
- [ ] Video upload extracts <= 64 frames and saves them in `backend/frames/<job_id>/`.
- [ ] Frontend polls and renders REAL / UNCERTAIN / FAKE verdict, probability meter, and frame gallery.
