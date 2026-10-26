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

# Images larger than this (longest side, px) are downscaled before MTCNN detection.
# The crop itself is always taken from the full-resolution image (same as backend inference).
DETECTION_MAX_SIDE = 1280


def build_mtcnn(device: torch.device) -> MTCNN:
    """MTCNN with the same configuration as backend inference."""
    return MTCNN(
        image_size=224,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        keep_all=True,
        post_process=False,
        device=device,
    )


def extract_padded_face(
    pil_image: Image.Image,
    mtcnn: MTCNN,
    padding: float = 0.2,
    max_side: int = DETECTION_MAX_SIDE,
) -> Optional[Image.Image]:
    """
    Detect faces using MTCNN, select largest face, apply 20% padding, and crop.
    Returns None if no face is detected (same behaviour as backend inference,
    which rejects images without a face).
    """
    img_w, img_h = pil_image.size

    # Downscale for faster detection, then map boxes back to full resolution
    scale = 1.0
    detect_img = pil_image
    if max(img_w, img_h) > max_side:
        scale = max_side / float(max(img_w, img_h))
        detect_img = pil_image.resize(
            (max(1, int(round(img_w * scale))), max(1, int(round(img_h * scale)))),
            Image.BILINEAR,
        )

    boxes, _probs = mtcnn.detect(detect_img)
    if boxes is None or len(boxes) == 0:
        return None
    boxes = boxes / scale

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
        return None

    # Apply 20% padding (same formula as backend inference)
    x1, y1, x2, y2 = largest_box
    w = x2 - x1
    h = y2 - y1

    pad_x = int(math.floor(w * padding))
    pad_y = int(math.floor(h * padding))

    new_x1 = max(0, int(round(x1 - pad_x)))
    new_y1 = max(0, int(round(y1 - pad_y)))
    new_x2 = min(img_w, int(round(x2 + pad_x)))
    new_y2 = min(img_h, int(round(y2 + pad_y)))

    if new_x2 > new_x1 and new_y2 > new_y1:
        return pil_image.crop((new_x1, new_y1, new_x2, new_y2))
    return None


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

    When extract_face=True, MTCNN face cropping runs ONCE per image in the main
    process and the crops are cached as lossless PNGs in cache_dir. Every epoch then
    reads the cached crops, so:
      - MTCNN no longer re-runs every epoch (big training speed-up)
      - DataLoader workers never touch CUDA (num_workers > 0 is safe)
      - Images without a detectable face are skipped instead of silently
        training on the full uncropped image.
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[Callable] = None,
        extract_face: bool = True,
        device: Optional[torch.device] = None,
        padding: float = 0.2,
        cache_dir: Optional[Path] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extract_face = extract_face
        self.padding = padding
        self.device = device or torch.device("cpu")
        self.cache_dir = Path(cache_dir) if cache_dir else self.root_dir.parent / "_face_cache" / self.root_dir.name

        self.samples: List[Tuple[Path, float]] = []
        self._load_samples()

        if self.extract_face:
            self.samples = self._build_face_cache(self.samples)

    def _load_samples(self) -> None:
        real_dir = self.root_dir / "real"
        fake_dir = self.root_dir / "fake"

        if real_dir.exists() and real_dir.is_dir():
            for p in sorted(real_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.samples.append((p, 0.0))  # 0.0 for REAL

        if fake_dir.exists() and fake_dir.is_dir():
            for p in sorted(fake_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.samples.append((p, 1.0))  # 1.0 for FAKE

    def _build_face_cache(self, samples: List[Tuple[Path, float]]) -> List[Tuple[Path, float]]:
        """Crop faces once and return samples pointing at the cached crops."""
        mtcnn = None
        cached: List[Tuple[Path, float]] = []
        skipped_no_face = 0
        skipped_errors = 0

        total = len(samples)
        for i, (img_path, label) in enumerate(samples, start=1):
            class_dir = "real" if label == 0.0 else "fake"
            crop_path = self.cache_dir / class_dir / f"{img_path.name}.png"
            no_face_marker = crop_path.with_suffix(".noface")

            if crop_path.exists():
                cached.append((crop_path, label))
                continue
            if no_face_marker.exists():
                skipped_no_face += 1
                continue

            if mtcnn is None:
                mtcnn = build_mtcnn(self.device)
                print(f"Extracting face crops for {self.root_dir} -> {self.cache_dir} (runs once)...")

            crop_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(img_path) as img:
                    face = extract_padded_face(img.convert("RGB"), mtcnn, padding=self.padding)
            except Exception as e:
                skipped_errors += 1
                print(f"  [WARN] Could not process {img_path}: {e}")
                continue

            if face is None:
                skipped_no_face += 1
                no_face_marker.touch()
                continue

            face.save(crop_path)
            cached.append((crop_path, label))

            if i % 500 == 0:
                print(f"  {i}/{total} images processed")

        if skipped_no_face or skipped_errors:
            print(
                f"  {self.root_dir.name}: skipped {skipped_no_face} image(s) with no detectable face "
                f"and {skipped_errors} unreadable image(s)."
            )

        # Free GPU memory used by MTCNN before training starts
        del mtcnn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return cached

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]

        # Read image (already a face crop if extract_face=True) using PIL in RGB mode
        with Image.open(img_path) as img:
            img = img.convert("RGB")

            if self.transform is not None:
                tensor = self.transform(img)
            else:
                tensor = transforms.ToTensor()(img)

        label_tensor = torch.tensor([label], dtype=torch.float32)
        return tensor, label_tensor
