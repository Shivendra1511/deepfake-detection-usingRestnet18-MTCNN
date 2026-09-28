import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from app.config import settings

class ResNet18Detector(nn.Module):
    """
    Fine-tuned ResNet18 Binary Classifier for Deepfake Detection.
    Backbone: ImageNet pretrained ResNet18.
    Classifier head:
      Dropout(p=0.5) -> Linear(512 -> 1) -> outputs single raw logit.
    No Sigmoid inside the forward pass to support BCEWithLogitsLoss during training.
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # ResNet18 has 512 features before the final linear layer
        in_features = self.backbone.fc.in_features  # 512

        # Replace standard 1000-class fc layer with Dropout + Linear(512 -> 1)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input: Tensor of shape (B, 3, 224, 224)
        Output: Tensor of shape (B, 1) containing raw logit(s)
        """
        return self.backbone(x)

def get_device() -> torch.device:
    """
    Automatically select CUDA GPU if available; otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Singleton cache for inference model instance
_CACHED_DETECTOR: Optional[ResNet18Detector] = None

def load_detector(model_path: Optional[str] = None, force_reload: bool = False) -> ResNet18Detector:
    """
    Load the trained ResNet18Detector checkpoint from disk.
    If detector.pt does not exist, raises FileNotFoundError with a clear message.
    """
    global _CACHED_DETECTOR

    if _CACHED_DETECTOR is not None and not force_reload:
        return _CACHED_DETECTOR

    target_path = Path(model_path if model_path else settings.MODEL_PATH).resolve()

    if not target_path.exists() or not target_path.is_file():
        raise FileNotFoundError(
            "Trained model not found. Please train the model and place detector.pt inside backend/data/."
        )

    device = get_device()
    # Instantiate architecture with pretrained=False since we load our fine-tuned weights
    model = ResNet18Detector(pretrained=False)

    try:
        checkpoint = torch.load(str(target_path), map_location=device)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Error loading model checkpoint from {target_path}: {str(e)}")

    model.to(device)
    model.eval()

    _CACHED_DETECTOR = model
    return _CACHED_DETECTOR

def classify_probability(probability: float) -> str:
    """
    Classify fake probability into REAL / UNCERTAIN / FAKE using configurable thresholds.
    """
    if probability < settings.LOWER_THRESHOLD:
        return "REAL"
    elif probability > settings.UPPER_THRESHOLD:
        return "FAKE"
    else:
        return "UNCERTAIN"

def predict_tensor(
    model: ResNet18Detector,
    tensor: torch.Tensor,
    device: Optional[torch.device] = None
) -> Tuple[float, str]:
    """
    Perform inference on a preprocessed face tensor (1, 3, 224, 224).
    Applies torch.sigmoid(logit) to obtain fake probability.
    Returns: (probability, label)
    """
    if device is None:
        device = get_device()

    if tensor.dim() == 3:
        # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
        tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()

    score = round(float(prob), 4)
    label = classify_probability(score)

    return score, label
