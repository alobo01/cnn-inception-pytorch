
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Sequence, Union
from utils.config import ModelConfig, ConvBlockConfig, ClassifierConfig
from implementations.classifier import Classifier


class ConvBlock(nn.Module):
    """
    A modular convolutional block built from ConvBlockConfig.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_bn: bool,
        pool_kernel_size: int,
        pool_stride: int,
        num_blocks: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(num_blocks):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # one pooling after all convs
        layers.append(nn.MaxPool2d(pool_kernel_size, pool_stride))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    @classmethod
    def from_config(
        cls,
        in_channels: int,
        cfg: ConvBlockConfig,
    ) -> ConvBlock:
        return cls(
            in_channels=in_channels,
            out_channels=cfg.filters,
            kernel_size=cfg.kernel_size,
            use_bn=cfg.use_bn,
            pool_kernel_size=cfg.pool_kernel_size,
            pool_stride=cfg.pool_stride,
            num_blocks=cfg.num_blocks,
        )


class StandardCNN(nn.Module):
    """
    Configurable CNN using a sequence of ConvBlock modules and a shared Classifier head.
    """
    def __init__(
        self,
        num_classes: int,
        conv_blocks: Sequence[ConvBlockConfig],
        classifier_cfg: ClassifierConfig,
        avg_pool2d: bool = True,
        input_shape: Sequence[int, int] = (256, 256),  # (H, W) of input images
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 3
        # Build feature extractor from config blocks
        for block_cfg in conv_blocks:
            block = ConvBlock.from_config(in_ch, block_cfg)
            layers.append(block)
            in_ch = block_cfg.filters

        self.features = nn.Sequential(*layers)
         # Choose pooling vs. flatten
        if avg_pool2d:
            # Global avg pool to 1x1 + flatten
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1)
            )
            classifier_in = in_ch
        else:
            # Flatten full feature map
            self.pool = nn.Identity()
            # Compute the flattened size after convolutions
            h, w = input_shape[0], input_shape[1]
            for block_cfg in conv_blocks:
                for _ in range(block_cfg.num_blocks):
                    h = h // block_cfg.pool_kernel_size
                    w = w // block_cfg.pool_kernel_size
            flattened_size = conv_blocks[-1].filters * h * w
            classifier_in = flattened_size

        # Shared classifier head
        self.classifier = Classifier.from_config(
            input_dim=classifier_in,
            num_classes=num_classes,
            cfg=classifier_cfg,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    @classmethod
    def build_from_config(
        cls,
        model_cfg: Union[ModelConfig, dict],
        num_classes: int
    ) -> StandardCNN:
        if isinstance(model_cfg, dict):
            model_cfg = ModelConfig(**model_cfg)
        return cls(
            num_classes=num_classes,
            conv_blocks=model_cfg.conv_blocks,
            classifier_cfg=model_cfg.classifier,
            avg_pool2d=model_cfg.avg_pool2d,
        )