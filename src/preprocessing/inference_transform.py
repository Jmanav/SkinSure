from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image into a normalised, batched tensor ready for SkinEnsemble.predict().

    Returns a tensor of shape (1, 3, 224, 224).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    return _transform(image).unsqueeze(0)
