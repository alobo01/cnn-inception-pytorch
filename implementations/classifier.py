from __future__ import annotations
import torch
import torch.nn as nn
from utils.config import ClassifierConfig


class Classifier(nn.Module):
    """
    Shared MLP classifier head constructed from a ClassifierConfig,
    expecting a pre-flattened input vector of size `input_dim`.
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        cfg: ClassifierConfig,
    ):
        super().__init__()
        layers: list[nn.Module] = []

        # # optional dropout at input
        # if cfg.dropout > 0:
        #     layers.append(nn.Dropout(cfg.dropout))

        Act = getattr(nn, cfg.activation, nn.ReLU)
        # If no hidden layers, direct Linear
        if not cfg.layers:
            layers.append(nn.Linear(input_dim, num_classes))
        else:
            # first hidden layer
            first_hidden = cfg.layers[0]
            layers.append(nn.Linear(input_dim, first_hidden))
            try:
                layers.append(Act(inplace=True))
            except TypeError:
                layers.append(Act())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))

            # subsequent hidden layers
            prev_dim = first_hidden
            for dim in cfg.layers[1:]:
                layers.append(nn.Linear(prev_dim, dim))
                try:
                    layers.append(Act(inplace=True))
                except TypeError:
                    layers.append(Act())
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
                prev_dim = dim

            # final output layer
            layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @classmethod
    def from_config(
        cls,
        input_dim: int,
        num_classes: int,
        cfg: ClassifierConfig,
    ) -> Classifier:
        return cls(input_dim, num_classes, cfg)