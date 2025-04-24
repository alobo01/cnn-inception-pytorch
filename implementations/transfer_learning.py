from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from torchvision import models

from implementations.classifier import Classifier
from utils.config import ClassifierConfig, ModelConfig


# --------------------------------------------------------------------------- #
#                                 helpers                                     #
# --------------------------------------------------------------------------- #
def load_vgg19_local(
    weights_path: str | os.PathLike = "vgg19.pth",
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """
    Loads a VGG-19 model with locally stored weights (pre-downloaded) and moves it to the specified device.

    Parameters
    ----------
    weights_path : str or PathLike
        Path to the .pth file containing the local weights.
    device : torch.device
        Device on which to load the model and weights.

    Returns
    -------
    nn.Module
        The VGG-19 model with loaded weights on `device`.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Local weights '{weights_path}' not found. Run the download script before training."
        )

    # Build the architecture without any pretrained weights
    model = models.vgg19(weights=None)
    # Load state_dict onto the right device
    state_dict = torch.load(weights_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # print("❗ Missing keys:", missing)
    # print("❗ Unexpected keys:", unexpected)
    return model.to(device)


# --------------------------------------------------------------------------- #
#                              main class                                     #
# --------------------------------------------------------------------------- #
class TransferLearningCNN(nn.Module):
    """
    Transfer Learning model based on VGG-19 with a fully configurable head.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        trainable_layers: int,
        weights_path: str,
        classifier_cfg: ClassifierConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device

        # 1) Load backbone on device
        base = load_vgg19_local(weights_path, device)

        # 2) Freeze/unfreeze conv-blocks
        if trainable_layers == 0:
            for p in base.features.parameters():
                p.requires_grad = False
        else:
            blocks: list[nn.Module] = []
            curr: list[nn.Module] = []
            for layer in base.features.children():
                curr.append(layer)
                if isinstance(layer, nn.MaxPool2d):
                    blocks.append(nn.Sequential(*curr))
                    curr = []
            for p in base.features.parameters():
                p.requires_grad = False
            for blk in blocks[-trainable_layers:]:
                for p in blk.parameters():
                    p.requires_grad = True

        self.features = base.features
        self.avgpool  = base.avgpool
        # Shared classifier
        in_feats = base.classifier[0].in_features
        self.classifier = Classifier.from_config(
            in_features=in_feats,
            num_classes=num_classes,
            cfg=classifier_cfg,
        )
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    @classmethod
    def build_from_config(
        cls,
        model_cfg: Union[ModelConfig, Dict[str, Any]],
        num_classes: int,
        device: torch.device | None = None,
    ) -> TransferLearningCNN:
        if isinstance(model_cfg, dict):
            model_cfg = ModelConfig(**model_cfg)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(
            num_classes=num_classes,
            trainable_layers=model_cfg.trainable_layers,
            weights_path=model_cfg.weights_path,
            classifier_cfg=model_cfg.classifier,
            device=device,
        )