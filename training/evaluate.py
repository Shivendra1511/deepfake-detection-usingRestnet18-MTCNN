import argparse
import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from model import ResNet18Detector, get_device
from dataset import DeepfakeDataset
from transforms import get_val_transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ResNet18 Deepfake Detection Model")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(CURRENT_DIR.parent / "dataset" / "val"),
        help="Path to validation dataset directory"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(CURRENT_DIR.parent / "backend" / "data" / "detector.pt"),
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(CURRENT_DIR.parent / "results" / "training" / "evaluation_report.json"),
        help="Output path for evaluation JSON metrics report"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device()

    print("=" * 65)
    print(" DEEPFAKE MODEL EVALUATION")
    print("=" * 65)
    print(f"Device: {device}")
    print(f"Model path: {args.model_path}")
    print(f"Validation dataset: {args.dataset_dir}")
    print("=" * 65)

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[ERROR] Trained model checkpoint not found at: {model_path}")
        print("Please train the model first by running: python train.py")
        sys.exit(1)

    val_dir = Path(args.dataset_dir)
    if not val_dir.exists():
        print(f"[ERROR] Validation directory not found at: {val_dir}")
        sys.exit(1)

    val_dataset = DeepfakeDataset(
        root_dir=val_dir,
        transform=get_val_transforms(image_size=224),
        extract_face=True,
        device=device
    )

    if len(val_dataset) == 0:
        print(f"[ERROR] Validation dataset at {val_dir} contains 0 images.")
        sys.exit(1)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Load Model Checkpoint
    model = ResNet18Detector(pretrained=False)
    checkpoint = torch.load(str(model_path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    print(f"\nEvaluating on {len(val_dataset)} validation samples...")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            true_labels = labels.cpu().numpy().flatten().astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(true_labels.tolist())

    avg_loss = total_loss / len(val_dataset)
    acc = accuracy_score(all_labels, all_preds) * 100.0
    prec = precision_score(all_labels, all_preds, zero_division=0) * 100.0
    rec = recall_score(all_labels, all_preds, zero_division=0) * 100.0
    f1 = f1_score(all_labels, all_preds, zero_division=0) * 100.0
    # labels=[0, 1] keeps the matrix 2x2 even if the set contains only one class
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1]).tolist()

    report = {
        "samples_evaluated": len(val_dataset),
        "validation_loss": round(avg_loss, 4),
        "accuracy_percent": round(acc, 2),
        "precision_percent": round(prec, 2),
        "recall_percent": round(rec, 2),
        "f1_score_percent": round(f1, 2),
        "confusion_matrix": {
            "true_real_pred_real (TN)": cm[0][0],
            "true_real_pred_fake (FP)": cm[0][1],
            "true_fake_pred_real (FN)": cm[1][0],
            "true_fake_pred_fake (TP)": cm[1][1]
        }
    }

    print("\n" + "=" * 45)
    print("          EVALUATION REPORT")
    print("=" * 45)
    print(f"  Samples:            {report['samples_evaluated']}")
    print(f"  Validation Loss:    {report['validation_loss']:.4f}")
    print(f"  Accuracy:           {report['accuracy_percent']:.2f}%")
    print(f"  Precision:          {report['precision_percent']:.2f}%")
    print(f"  Recall:             {report['recall_percent']:.2f}%")
    print(f"  F1 Score:           {report['f1_score_percent']:.2f}%")
    print("-" * 45)
    print("  Confusion Matrix:")
    print(f"    [[TN: {cm[0][0]}, FP: {cm[0][1]}],")
    print(f"     [FN: {cm[1][0]}, TP: {cm[1][1]}]]")
    print("=" * 45)

    out_json_path = Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nReport saved to: {out_json_path.resolve()}\n")

if __name__ == "__main__":
    main()
