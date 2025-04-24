from __future__ import annotations
import torch
import torch.nn as nn
from utils.config import ClassifierConfig

class Classifier(nn.Module):
    """
    Shared MLP classifier head constructed from a ClassifierConfig.
    """
    def __init__(self, in_features: int, num_classes: int, cfg: ClassifierConfig):
        super().__init__()
        layers: list[nn.Module] = []
        Act = getattr(nn, cfg.activation, nn.ReLU)
        dims = [in_features, *cfg.layers, num_classes]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                # hidden layer: activation + optional dropout
                try:
                    layers.append(Act(inplace=True))
                except TypeError:
                    layers.append(Act())
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @classmethod
    def from_config(
        cls,
        in_features: int,
        num_classes: int,
        cfg: ClassifierConfig
    ) -> Classifier:
        return cls(in_features, num_classes, cfg)