from torchvision import transforms
from PIL import Image
import numpy as np

# ImageNet standard normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomGamma:
    """
    TRAINING-ONLY: random gamma correction for lighting variation.
    gamma is sampled uniformly from [0.8, 1.2].
    """

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


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Data augmentation transforms applied ONLY during model training:
      - Random horizontal flip
      - Brightness / small color changes (ColorJitter)
      - Gamma variation
      - Conversion to tensor [0, 1]
      - ImageNet standard normalization
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
        ),
        RandomGamma(gamma_range=(0.8, 1.2), p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Deterministic transforms applied during validation, evaluation, and inference:
      - Resize to target (224x224)
      - Conversion to tensor [0, 1]
      - ImageNet standard normalization
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
