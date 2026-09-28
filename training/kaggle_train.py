"""
Kaggle training script for ResNet18 + MTCNN Deepfake Detection.

This file is self-contained so you can upload it to a Kaggle Notebook
without copying the rest of the repo.

HOW TO USE ON KAGGLE
--------------------
1. Create a new Kaggle Notebook.
2. Accelerator: GPU (T4 or P100).
3. Add your dataset (Add Input). Folder layout must be:

       train/real/
       train/fake/
       val/real/
       val/fake/

   If your zip has an extra parent folder, set DATASET_DIR below to that folder.

4. Copy this file into a notebook cell, OR File -> Upload this .py as a dataset
   and run it from a cell:

       !python /kaggle/input/<this-script-dataset>/kaggle_train.py

5. After training, download from Output:
       /kaggle/working/detector.pt

6. On your laptop, copy detector.pt to:
       backend/data/detector.pt
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# 0. CONFIG — edit these for your Kaggle dataset
# ---------------------------------------------------------------------------

# Path to the folder that CONTAINS train/ and val/
# Example after attaching a Kaggle dataset named "deepfake-faces":
#   DATASET_DIR = Path("/kaggle/input/deepfake-faces")
# If the zip has a nested folder:
#   DATASET_DIR = Path("/kaggle/input/deepfake-faces/dataset")
DATASET_DIR = Path("/kaggle/input/YOUR_DATASET_NAME")

OUTPUT_DIR = Path("/kaggle/working")
MODEL_PATH = OUTPUT_DIR / "detector.pt"
RESULTS_DIR = OUTPUT_DIR / "training_plots"
EVAL_JSON_PATH = OUTPUT_DIR / "evaluation_report.json"

EPOCHS = 10
BATCH_SIZE = 32          # Kaggle GPU can usually handle 32; drop to 16 if OOM
LEARNING_RATE = 1e-4     # documented default, not a magic number
WEIGHT_DECAY = 1e-4
SEED = 42
IMAGE_SIZE = 224
EXTRACT_FACE = True      # must match backend inference (MTCNN + 20% padding)
FACE_PADDING = 0.2
NUM_WORKERS = 2          # Kaggle Linux; 0 if DataLoader hangs

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def install_missing_packages() -> None:
    """Install packages Kaggle images often lack. Torch/torchvision are preinstalled."""
    try:
        import facenet_pytorch  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "facenet-pytorch"]
        )


def resolve_dataset_dir(configured: Path) -> Path:
    """
    Accept either:
      /kaggle/input/<name>/train/...
    or a nested folder that contains train/ and val/.
    """
    if (configured / "train").is_dir() and (configured / "val").is_dir():
        return configured

    if configured.exists():
        for child in configured.iterdir():
            if child.is_dir() and (child / "train").is_dir() and (child / "val").is_dir():
                print(f"Using nested dataset folder: {child}")
                return child

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        print("Configured DATASET_DIR not found. Scanning /kaggle/input ...")
        for ds in sorted(kaggle_input.iterdir()):
            if not ds.is_dir():
                continue
            if (ds / "train").is_dir() and (ds / "val").is_dir():
                print(f"Auto-detected dataset at: {ds}")
                return ds
            for child in ds.iterdir():
                if child.is_dir() and (child / "train").is_dir() and (child / "val").is_dir():
                    print(f"Auto-detected nested dataset at: {child}")
                    return child

    raise FileNotFoundError(
        f"Could not find train/ and val/ under {configured}. "
        "Add your Kaggle dataset and set DATASET_DIR to the folder that contains train/ and val/."
    )


# ---------------------------------------------------------------------------
# 1. Model — identical to training/model.py and backend ResNet18Detector
# ---------------------------------------------------------------------------

def build_model_code():
    import torch
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    class ResNet18Detector(nn.Module):
        """ImageNet ResNet18 + Dropout(0.5) + Linear(512 -> 1) logit."""

        def __init__(self, pretrained: bool = True):
            super().__init__()
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
            in_features = self.backbone.fc.in_features  # 512
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, 1),
            )

        def forward(self, x):
            return self.backbone(x)

    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return ResNet18Detector, get_device


# ---------------------------------------------------------------------------
# 2. Transforms — same ImageNet norm as local training/inference
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 224):
    from PIL import Image
    import numpy as np
    from torchvision import transforms

    class RandomGamma:
        def __init__(self, gamma_range=(0.8, 1.2), p: float = 0.5):
            self.gamma_range = gamma_range
            self.p = p

        def __call__(self, img: Image.Image) -> Image.Image:
            if np.random.rand() > self.p:
                return img
            gamma = float(np.random.uniform(self.gamma_range[0], self.gamma_range[1]))
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = np.clip(arr ** gamma, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
            return Image.fromarray(arr)

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1, hue=0.05),
        RandomGamma(gamma_range=(0.8, 1.2), p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224):
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# 3. Dataset — MTCNN largest face + 20% padding (same as local training)
# ---------------------------------------------------------------------------

class DeepfakeDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        transform: Optional[Callable] = None,
        extract_face: bool = True,
        device=None,
        padding: float = 0.2,
    ):
        import torch
        from facenet_pytorch import MTCNN

        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extract_face = extract_face
        self.padding = padding
        self.device = device or torch.device("cpu")
        self.samples: List[Tuple[Path, float]] = []
        self._load_samples()
        self.mtcnn = None
        if self.extract_face:
            self.mtcnn = MTCNN(
                image_size=224,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                keep_all=True,
                post_process=False,
                device=self.device,
            )

    def _load_samples(self) -> None:
        real_dir = self.root_dir / "real"
        fake_dir = self.root_dir / "fake"
        if real_dir.is_dir():
            for p in real_dir.iterdir():
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.samples.append((p, 0.0))
        if fake_dir.is_dir():
            for p in fake_dir.iterdir():
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.samples.append((p, 1.0))

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_padded_face(self, pil_image):
        if self.mtcnn is None:
            return pil_image
        try:
            boxes, _probs = self.mtcnn.detect(pil_image)
            if boxes is None or len(boxes) == 0:
                return pil_image
            largest_box = None
            max_area = -1.0
            for box in boxes:
                x1, y1, x2, y2 = box
                area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
                if area > max_area:
                    max_area = area
                    largest_box = box
            if largest_box is None:
                return pil_image
            img_w, img_h = pil_image.size
            x1, y1, x2, y2 = largest_box
            pad_x = int(math.floor((x2 - x1) * self.padding))
            pad_y = int(math.floor((y2 - y1) * self.padding))
            new_x1 = max(0, int(round(x1 - pad_x)))
            new_y1 = max(0, int(round(y1 - pad_y)))
            new_x2 = min(img_w, int(round(x2 + pad_x)))
            new_y2 = min(img_h, int(round(y2 + pad_y)))
            if new_x2 > new_x1 and new_y2 > new_y1:
                return pil_image.crop((new_x1, new_y1, new_x2, new_y2))
            return pil_image
        except Exception:
            return pil_image

    def __getitem__(self, idx: int):
        from PIL import Image
        import torch
        from torchvision import transforms as T

        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.extract_face:
                img = self._extract_padded_face(img)
            tensor = self.transform(img) if self.transform is not None else T.ToTensor()(img)
        return tensor, torch.tensor([label], dtype=torch.float32)


# ---------------------------------------------------------------------------
# 4. Utils
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_training_curves(history, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], "b-o", label="Training Loss")
    plt.title("Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_loss"], "r-o", label="Validation Loss")
    plt.title("Validation Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "validation_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_acc"], "g-o", label="Validation Accuracy (%)")
    plt.title("Validation Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "validation_accuracy.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# 5. Train + evaluate
# ---------------------------------------------------------------------------

def train():
    import torch.nn as nn
    from torch.utils.data import DataLoader

    install_missing_packages()
    set_seed(SEED)

    ResNet18Detector, get_device = build_model_code()
    device = get_device()
    dataset_root = resolve_dataset_dir(DATASET_DIR)
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    print("=" * 65)
    print(" KAGGLE TRAINING — ResNet18 + MTCNN Deepfake Detector")
    print("=" * 65)
    print(f"Device:          {device}")
    print(f"Dataset:         {dataset_root}")
    print(f"Epochs:          {EPOCHS}")
    print(f"Batch size:      {BATCH_SIZE}")
    print(f"Learning rate:   {LEARNING_RATE}")
    print(f"Weight decay:    {WEIGHT_DECAY}")
    print(f"Extract faces:   {EXTRACT_FACE}")
    print(f"Checkpoint:      {MODEL_PATH}")
    print("=" * 65)

    train_dataset = DeepfakeDataset(
        root_dir=train_dir,
        transform=get_train_transforms(IMAGE_SIZE),
        extract_face=EXTRACT_FACE,
        device=device,
        padding=FACE_PADDING,
    )
    val_dataset = DeepfakeDataset(
        root_dir=val_dir,
        transform=get_val_transforms(IMAGE_SIZE),
        extract_face=EXTRACT_FACE,
        device=device,
        padding=FACE_PADDING,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            f"Empty dataset. Train={len(train_dataset)} Val={len(val_dataset)}. "
            "Need images in train/real, train/fake, val/real, val/fake."
        )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = ResNet18Detector(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = -1.0
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)
        epoch_train_loss = running_train_loss / len(train_dataset)

        model.eval()
        running_val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                running_val_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc = (correct / len(val_dataset)) * 100.0
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:6.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "resnet18",
                    "image_size": IMAGE_SIZE,
                    "val_acc": best_val_acc,
                    "val_loss": epoch_val_loss,
                    "epoch": epoch,
                },
                str(MODEL_PATH),
            )
            print(f"  >>> Best checkpoint saved ({best_val_acc:.2f}%) -> {MODEL_PATH}")

    plot_training_curves(history, RESULTS_DIR)
    print(f"\nBest val acc: {best_val_acc:.2f}%")
    print(f"Model: {MODEL_PATH}")
    print(f"Plots: {RESULTS_DIR}")
    print("Download detector.pt from Kaggle Output and copy to backend/data/detector.pt")
    return MODEL_PATH, val_dir, device


def evaluate(model_path: Path, val_dir: Path, device) -> None:
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    ResNet18Detector, _ = build_model_code()
    val_dataset = DeepfakeDataset(
        root_dir=val_dir,
        transform=get_val_transforms(IMAGE_SIZE),
        extract_face=EXTRACT_FACE,
        device=device,
        padding=FACE_PADDING,
    )
    loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = ResNet18Detector(pretrained=False)
    checkpoint = torch.load(str(model_path), map_location=device)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * images.size(0)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend((probs >= 0.5).astype(int).tolist())
            all_labels.extend(labels.cpu().numpy().flatten().astype(int).tolist())

    cm = confusion_matrix(all_labels, all_preds).tolist()
    report = {
        "samples_evaluated": len(val_dataset),
        "validation_loss": round(total_loss / len(val_dataset), 4),
        "accuracy_percent": round(accuracy_score(all_labels, all_preds) * 100.0, 2),
        "precision_percent": round(precision_score(all_labels, all_preds, zero_division=0) * 100.0, 2),
        "recall_percent": round(recall_score(all_labels, all_preds, zero_division=0) * 100.0, 2),
        "f1_score_percent": round(f1_score(all_labels, all_preds, zero_division=0) * 100.0, 2),
        "confusion_matrix": cm,
    }
    print("\nEVALUATION")
    print(json.dumps(report, indent=2))
    with open(EVAL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"Saved {EVAL_JSON_PATH}")


if __name__ == "__main__":
    saved_model, val_folder, used_device = train()
    evaluate(saved_model, val_folder, used_device)
