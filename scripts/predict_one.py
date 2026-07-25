from __future__ import annotations

import argparse
import logging

from PIL import Image

from src.models.ensemble import SkinEnsemble
from src.preprocessing.inference_transform import preprocess_image
from src.utils.config import get_checkpoint_paths, get_model_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ensemble inference on a single image.")
    parser.add_argument("image", type=str, help="Path to the input image.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    paths = get_checkpoint_paths()
    weights_cfg = get_model_config().weights
    weights = (weights_cfg["efficientnet"], weights_cfg["convnext"], weights_cfg["swin"])

    ensemble = SkinEnsemble(paths, weights=weights)

    image = Image.open(args.image)
    image_tensor = preprocess_image(image)

    predictions = ensemble.predict(image_tensor)

    print(f"Primary   : {predictions.primary_label} ({predictions.primary_confidence:.3f})")
    print(f"Secondary : {predictions.secondary_label} ({predictions.secondary_confidence:.3f})")
    print(f"Agreement : {predictions.agreement_score:.2f} ({predictions.confidence_tier()})")


if __name__ == "__main__":
    main()
