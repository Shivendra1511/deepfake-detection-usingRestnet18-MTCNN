# Deepfake Dataset Directory

Place your training and validation images into the corresponding directories below:

- `dataset/train/real/`: Authentic, real human face images (Label: 0)
- `dataset/train/fake/`: Deepfake or manipulated face images (Label: 1)
- `dataset/val/real/`: Real validation images
- `dataset/val/fake/`: Deepfake validation images

### Supported Formats
- `.jpg`, `.jpeg`, `.png`, `.webp`

The training pipeline automatically reads from these folders and runs MTCNN face extraction (with 20% padding) before passing crops to ResNet18.
