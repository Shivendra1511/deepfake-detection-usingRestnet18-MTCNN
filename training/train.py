import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add training directory to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from model import ResNet18Detector, get_device
from dataset import DeepfakeDataset
from transforms import get_train_transforms, get_val_transforms
from utils import set_seed, plot_training_curves

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet18 Deepfake Detection Binary Classifier"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(CURRENT_DIR.parent / "dataset"),
        help="Path to dataset directory containing train/ and val/ folders"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and validation (default: 16)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for Adam optimizer (default: 1e-4)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 penalty) for Adam (default: 1e-4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default=str(CURRENT_DIR.parent / "backend" / "data" / "detector.pt"),
        help="Destination path for best trained model checkpoint"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(CURRENT_DIR.parent / "results" / "training"),
        help="Directory to save training metric graphs"
    )
    parser.add_argument(
        "--extract_face",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run MTCNN face extraction on dataset images (default: enabled). Use --no-extract_face to disable."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where cached face crops are stored (default: <dataset_dir>/_face_cache). Delete it to re-extract faces."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print("=" * 65)
    print(" DEEPFAKE DETECTION MODEL TRAINING (ResNet18 + MTCNN)")
    print("=" * 65)
    print(f"Device: {device}")
    print(f"Dataset root: {args.dataset_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Initial Learning Rate: {args.lr}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Output Checkpoint: {args.output_model}")
    print("=" * 65)

    dataset_path = Path(args.dataset_dir)
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"

    if not train_dir.exists() or not val_dir.exists():
        print(f"\n[ERROR] Dataset folders not found at: {dataset_path}")
        print("Please ensure the following directory layout exists:")
        print("  dataset/train/real/")
        print("  dataset/train/fake/")
        print("  dataset/val/real/")
        print("  dataset/val/fake/")
        sys.exit(1)

    # 1. Initialize Datasets & DataLoaders
    print("\n[1/6] Loading datasets...")
    cache_root = Path(args.cache_dir) if args.cache_dir else dataset_path / "_face_cache"
    train_dataset = DeepfakeDataset(
        root_dir=train_dir,
        transform=get_train_transforms(image_size=224),
        extract_face=args.extract_face,
        device=device,
        cache_dir=cache_root / "train"
    )
    val_dataset = DeepfakeDataset(
        root_dir=val_dir,
        transform=get_val_transforms(image_size=224),
        extract_face=args.extract_face,
        device=device,
        cache_dir=cache_root / "val"
    )

    if len(train_dataset) == 0:
        print("\n[ERROR] Training dataset is empty.")
        print(f"Please add training images into '{train_dir / 'real'}' and '{train_dir / 'fake'}'.")
        sys.exit(1)

    if len(val_dataset) == 0:
        print("\n[ERROR] Validation dataset is empty.")
        print(f"Please add validation images into '{val_dir / 'real'}' and '{val_dir / 'fake'}'.")
        sys.exit(1)

    print(f"Found {len(train_dataset)} training samples.")
    print(f"Found {len(val_dataset)} validation samples.")

    # Number of workers: 0 is safest for Windows multiprocessing
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # 2. Build Model
    print("\n[2/6] Loading ImageNet pretrained ResNet18 and configuring binary head...")
    model = ResNet18Detector(pretrained=True)
    model.to(device)

    # 3. Loss & Optimizer
    # Binary Cross Entropy with Logits: expects raw logit and float targets (0.0 or 1.0)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 4. Learning Rate Scheduler (ReduceLROnPlateau: factor=0.5, patience=1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1
    )

    # History tracking for curves
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_val_acc = -1.0
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n[3/6] Starting training loop...")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        # --- Training Phase ---
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

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        correct_predictions = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)
                running_val_loss += loss.item() * images.size(0)

                # Predictions: probability >= 0.5 is FAKE (1.0), else REAL (0.0)
                probabilities = torch.sigmoid(logits)
                preds = (probabilities >= 0.5).float()
                correct_predictions += (preds == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc = (correct_predictions / len(val_dataset)) * 100.0

        current_lr = optimizer.param_groups[0]["lr"]

        # Step LR scheduler based on validation loss
        scheduler.step(epoch_val_loss)

        # Record metrics
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:6.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Save best model checkpoint
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "model_name": "resnet18",
                "image_size": 224,
                "val_acc": best_val_acc,
                "val_loss": epoch_val_loss,
                "epoch": epoch
            }
            torch.save(checkpoint_payload, str(output_path))
            print(f"  >>> Best model saved! (Validation Accuracy: {best_val_acc:.2f}%)")

    print("-" * 65)
    print("\n[4/6] Training complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {output_path.resolve()}")

    # 5. Plot training and validation graphs
    print("\n[5/6] Generating metric graphs...")
    results_dir = Path(args.results_dir)
    plot_training_curves(history, results_dir)
    print(f"Saved metric plots to: {results_dir.resolve()}")

    print("\n[6/6] Ready for Inference!")
    print("Start the FastAPI backend and React frontend to detect deepfakes in images and videos.")
    print("=" * 65)

if __name__ == "__main__":
    main()
