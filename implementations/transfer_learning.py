from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Sequence, Union

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
    model.load_state_dict(state_dict, strict=False)
    return model.to(device)

# Reduce gradient in pre-trained model
def scale_grad_hook(grad):
    return grad * 0.01

# --------------------------------------------------------------------------- #
#                              main class                                     #
# --------------------------------------------------------------------------- #
class TransferLearningCNN(nn.Module):
    """
    Transfer Learning model based on VGG-19 with a fully configurable head.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    trainable_layers : int
        How many of the last VGG conv-blocks to leave trainable (0 = freeze all).
    weights_path : str
        Path to the locally stored VGG-19 weights.
    classifier_layers : List[int]
        Hidden layer sizes for the classifier head (before final output).
    classifier_dropout : float
        Dropout probability between classifier layers.
    classifier_activation : str
        Name of the activation for hidden layers (e.g. "relu").
    device : torch.device
        Device (CPU or CUDA) to run the model on.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        trainable_layers: int,
        weights_path: str,
        classifier_cfg: ClassifierConfig,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        image_shape: Sequence[int, int] = (256, 256),
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
            # Split features into blocks (ending at each MaxPool2d)
            blocks: list[nn.Module] = []
            curr: list[nn.Module] = []
            for layer in base.features.children():
                curr.append(layer)
                if isinstance(layer, nn.MaxPool2d):
                    blocks.append(nn.Sequential(*curr))
                    curr = []
            # Freeze all first
            for p in base.features.parameters():
                p.requires_grad = False
            # Unfreeze last N blocks
            for blk in blocks[-trainable_layers:]:
                for p in blk.parameters():
                    p.requires_grad = True
                    p.register_hook(scale_grad_hook)

        # Attach feature extractor and pooling
        self.features = base.features
        self.avgpool = base.avgpool

        # after self.features and self.avgpool are set up:
        with torch.no_grad():
            # assume classifier_cfg.image_size is the H=W of your input
            H, W = image_shape[0], image_shape[1]
            dummy = torch.zeros(1, 3, H, W, device=self.device)
            feat = self.features(dummy)
            pooled = self.avgpool(feat)               # shape (1, C, Ph, Pw)
            classifier_in = int(pooled.numel() / 1)   # total features per example

        # 3) Build classifier head dynamically
        # Shared classifier head
        self.classifier = Classifier.from_config(
            input_dim=classifier_in,
            num_classes=num_classes,
            cfg=classifier_cfg,
        )

        # Move everything to device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ensure input on correct device
        x = x.to(self.device)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    

    @classmethod
    def build_from_config(
        cls,
        model_cfg: Union[ModelConfig, dict],
        num_classes: int
    ) -> TransferLearningCNN:
        if isinstance(model_cfg, dict):
            model_cfg = ModelConfig(**model_cfg)
        return cls(
            num_classes = num_classes,
            trainable_layers = model_cfg.trainable_layers,
            weights_path = model_cfg.weights_path,
            classifier_cfg = model_cfg.classifier
        )