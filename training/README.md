# Model Training Pipeline (ResNet18 + MTCNN)

This folder contains the training and evaluation scripts for the Deepfake Detection System based on transfer learning with **ResNet18** and **MTCNN face detection**.

---

## 1. Dataset Directory Layout

Place your dataset images inside the `dataset/` directory:

```text
dataset/
├── train/
│   ├── real/    # Real face images (labeled as 0.0)
│   └── fake/    # Manipulated / Deepfake face images (labeled as 1.0)
└── val/
    ├── real/    # Real face validation images
    └── fake/    # Manipulated face validation images
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`.

---

## 2. Training the Model

To train the model using default settings (10 epochs, batch size 16, lr 1e-4):

```powershell
cd d:\prpe\deepfake\training
python train.py
```

### Optional Arguments

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 10 | Total training epochs |
| `--batch_size` | 16 | Batch size |
| `--lr` | 0.0001 | Initial learning rate for Adam |
| `--weight_decay` | 0.0001 | Weight decay (L2 regularization) |
| `--seed` | 42 | Random seed for reproducibility |
| `--output_model` | `../backend/data/detector.pt` | Destination for best model checkpoint |
| `--results_dir` | `../results/training` | Where loss/accuracy graphs are saved |

Example with custom parameters:

```powershell
python train.py --epochs 15 --batch_size 32 --lr 0.0002
```

---

## 3. Training Process & Outputs

During training:
1. Pretrained **ResNet18** weights are loaded from ImageNet.
2. The final 1000-class layer is replaced with `Dropout(p=0.5)` + `Linear(512, 1)` outputting a single logit.
3. Training augmentations (random flip, color jitter, rotation) are applied.
4. Validation is evaluated every epoch using `BCEWithLogitsLoss`.
5. Learning rate scheduler (`ReduceLROnPlateau`, factor 0.5, patience 1) monitors validation loss.
6. The best performing model checkpoint is automatically saved to:
   ```text
   backend/data/detector.pt
   ```
7. Training and validation loss/accuracy curves are plotted and saved to:
   ```text
   results/training/training_loss.png
   results/training/validation_loss.png
   results/training/validation_accuracy.png
   ```

---

## 4. Evaluating the Model

After training, you can run standalone evaluation to calculate validation metrics (Accuracy, Precision, Recall, F1-Score, and Confusion Matrix):

```powershell
python evaluate.py
```

Results will be printed in a formatted table and saved to `results/training/evaluation_report.json`.

---

## 5. Training on Kaggle (recommended if your laptop is slow)

Use `kaggle_train.py` (self-contained; does not need the rest of the repo on Kaggle).

1. Zip your images with this layout (same as local training):

```text
train/real/
train/fake/
val/real/
val/fake/
```

2. Upload that zip as a **Kaggle Dataset**.
3. New Notebook → **GPU** (T4 or P100) → **Add Input** (your dataset).
4. Set `DATASET_DIR` near the top of `kaggle_train.py` to:

```python
DATASET_DIR = Path("/kaggle/input/YOUR_DATASET_NAME")
```

5. Paste the script into one notebook cell and run, **or** upload `kaggle_train.py` as a small dataset and run:

```python
!python /kaggle/input/<script-dataset>/kaggle_train.py
```

6. After it finishes, download **Output** file `detector.pt`.
7. Copy it onto your laptop:

```powershell
copy detector.pt d:\prpe\deepfake\backend\data\detector.pt
```

The checkpoint format matches local training:

```text
{
  "model_state_dict": ...,
  "model_name": "resnet18",
  "image_size": 224,
  ...
}
```

Install `facenet-pytorch` is done automatically if Kaggle does not already have it.
