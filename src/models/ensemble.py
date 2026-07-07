from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from api.schemas.predictions import Predictions
from src.models.convnext import ConvNeXtSkinClassifier, load_convnext_tiny
from src.models.efficientnet import EfficientSkinClassifier, load_efficientnet_b1
from src.models.swin import SwinSkinClassifier, load_swin_tiny

logger = logging.getLogger(__name__)

# Union type for any of the three backbone classifiers
_AnyModel = EfficientSkinClassifier | ConvNeXtSkinClassifier | SwinSkinClassifier


@dataclass(frozen=True)
class CheckpointPaths:
    efficientnet: str | Path
    convnext: str | Path
    swin: str | Path

class SkinEnsemble:
    """
    Weighted-average ensemble of EfficientNet-B1, ConvNeXt-Tiny, and Swin-Tiny.

    All three models must be trained on the same 5-class label set:
        Melanoma | Basal_Cell_Carcinoma | Tinea | Eczema | Keratosis

    agreement_score definition:
        Fraction of the three models whose argmax matches the ensemble argmax.
        0.33 → only 1 model agrees, 0.67 → 2 agree, 1.0 → all three agree.
    """

    def __init__(
        self,
        paths: CheckpointPaths,
        device: torch.device | None = None,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading ensemble models onto %s", self.device)
        self._eff,  classes_eff  = load_efficientnet_b1(str(paths.efficientnet), self.device)
        self._cnxt, classes_cnxt = load_convnext_tiny(str(paths.convnext),      self.device)
        self._swin, classes_swin = load_swin_tiny(str(paths.swin),              self.device)

        # Guard: all three checkpoints must share the same class list in the same order
        if not (classes_eff == classes_cnxt == classes_swin):
            raise ValueError(
                "Class list mismatch across checkpoints.\n"
                f"  EfficientNet : {classes_eff}\n"
                f"  ConvNeXt     : {classes_cnxt}\n"
                f"  Swin         : {classes_swin}"
            )
        self.classes: list[str] = classes_eff

        total = sum(weights)
        self._weights = tuple(w / total for w in weights)
        logger.info(
            "Ensemble ready | classes=%s | weights=(eff=%.2f, cnxt=%.2f, swin=%.2f)",
            self.classes,
            *self._weights,
        )

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Predictions:
        """
        Run ensemble inference on a single pre-processed image tensor.

        Args:
            image_tensor: Shape (1, 3, 224, 224), normalised with ImageNet stats,
                          already on CPU — this method moves it to self.device.

        Returns:
            Predictions with primary/secondary labels, confidences, and agreement_score.
        """
        x = image_tensor.to(self.device, memory_format=torch.channels_last)

        # Each model returns raw logits → convert to probabilities
        probs_eff  = F.softmax(self._eff(x),  dim=1)
        probs_cnxt = F.softmax(self._cnxt(x), dim=1)
        probs_swin = F.softmax(self._swin(x), dim=1)

        w_eff, w_cnxt, w_swin = self._weights
        ensemble_probs = (
            w_eff  * probs_eff
            + w_cnxt * probs_cnxt
            + w_swin * probs_swin
        )  # shape: (1, num_classes)

        ensemble_probs = ensemble_probs.squeeze(0)  # (num_classes,)

        # Top-2 predictions from the ensemble
        top2_conf, top2_idx = ensemble_probs.topk(2)
        primary_idx   = top2_idx[0].item()
        secondary_idx = top2_idx[1].item()

        primary_conf   = top2_conf[0].item()
        secondary_conf = top2_conf[1].item()

        # agreement_score: fraction of individual models whose argmax matches ensemble argmax
        individual_tops = [
            probs_eff.argmax(dim=1).item(),
            probs_cnxt.argmax(dim=1).item(),
            probs_swin.argmax(dim=1).item(),
        ]
        agreement_score = sum(t == primary_idx for t in individual_tops) / 3.0

        logger.debug(
            "Ensemble | primary=%s (%.3f) | secondary=%s (%.3f) | agreement=%.2f",
            self.classes[primary_idx],
            primary_conf,
            self.classes[secondary_idx],
            secondary_conf,
            agreement_score,
        )

        return Predictions(
            primary_label=self.classes[primary_idx],
            primary_confidence=primary_conf,
            secondary_label=self.classes[secondary_idx],
            secondary_confidence=secondary_conf,
            agreement_score=agreement_score,
        )
