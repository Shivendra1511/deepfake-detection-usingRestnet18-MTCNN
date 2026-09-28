import random
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed: int = 42) -> None:
    """
    Ensure experiment reproducibility by setting seeds for Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_training_curves(history: Dict[str, List[float]], output_dir: Path) -> None:
    """
    Generate and save actual training and validation metric graphs into output_dir.
    Saves:
      - training_loss.png
      - validation_loss.png
      - validation_accuracy.png
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # 1. Training Loss Curve
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

    # 2. Validation Loss Curve
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

    # 3. Validation Accuracy Curve
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
