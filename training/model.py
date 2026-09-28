import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Detector(nn.Module):
    """
    Fine-tuned ResNet18 Binary Classifier for Deepfake Detection.
    Backbone: ImageNet pretrained ResNet18.
    Classifier head:
      Dropout(p=0.5) -> Linear(512 -> 1) -> outputs single raw logit.
    No Sigmoid inside the forward pass to support BCEWithLogitsLoss during training.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # ResNet18 fc layer maps 512 in_features -> 1000 classes
        in_features = self.backbone.fc.in_features  # 512

        # Replace final classification layer with Dropout + Linear(512 -> 1)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input: Tensor of shape (B, 3, 224, 224)
        Output: Tensor of shape (B, 1) raw logit
        """
        return self.backbone(x)

def get_device() -> torch.device:
    """
    Automatically select CUDA GPU if available; otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
