from pathlib import Path
from typing import List, Tuple, Optional, Callable
import math
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from facenet_pytorch import MTCNN

# Supported image file extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for Deepfake Detection.
    Expects directory structure:
      root_dir/
        real/
        fake/
    Labels:
      REAL = 0.0
      FAKE = 1.0
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[Callable] = None,
        extract_face: bool = True,
        device: Optional[torch.device] = None,
        padding: float = 0.2,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extract_face = extract_face
        self.padding = padding
        self.device = device or torch.device("cpu")

        self.samples: List[Tuple[Path, float]] = []
        self._load_samples()

        # Initialize MTCNN for dataset face extraction if enabled
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

        if real_dir.exists() and real_dir.is_dir():
            for p in real_dir.iterdir():
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.samples.append((p, 0.0))  # 0.0 for REAL

        if fake_dir.exists() and fake_dir.is_dir():
            for p in fake_dir.iterdir():
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.samples.append((p, 1.0))  # 1.0 for FAKE

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_padded_face(self, pil_image: Image.Image) -> Image.Image:
        """
        Detect faces using MTCNN, select largest face, apply 20% padding, and crop.
        Falls back to original image if no face detected (keeps training sample usable).
        """
        if self.mtcnn is None:
            return pil_image

        try:
            boxes, probs = self.mtcnn.detect(pil_image)
            if boxes is None or len(boxes) == 0:
                return pil_image

            # Select largest face by area
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

            # Apply 20% padding (same formula as backend inference)
            img_w, img_h = pil_image.size
            x1, y1, x2, y2 = largest_box
            w = x2 - x1
            h = y2 - y1

            pad_x = int(math.floor(w * self.padding))
            pad_y = int(math.floor(h * self.padding))

            new_x1 = max(0, int(round(x1 - pad_x)))
            new_y1 = max(0, int(round(y1 - pad_y)))
            new_x2 = min(img_w, int(round(x2 + pad_x)))
            new_y2 = min(img_h, int(round(y2 + pad_y)))

            if new_x2 > new_x1 and new_y2 > new_y1:
                return pil_image.crop((new_x1, new_y1, new_x2, new_y2))
            return pil_image
        except Exception:
            return pil_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]

        # Read image using PIL in RGB mode
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.extract_face:
                img = self._extract_padded_face(img)

            if self.transform is not None:
                tensor = self.transform(img)
            else:
                tensor = transforms.ToTensor()(img)

        label_tensor = torch.tensor([label], dtype=torch.float32)
        return tensor, label_tensor
