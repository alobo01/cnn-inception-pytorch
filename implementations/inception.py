"""InceptionNet — fully YAML‑configurable implementation
=====================================================
This version exposes **every architectural knob** through a single
`build_from_config(cfg, num_classes)` factory so that the entire network
can be driven from a YAML file like the example shown earlier.

Key additions
-------------
* `InceptionBlock` now accepts `se_reduction`, `dropout_p`, and
  `bn_before_act` flags.
* A tidy `FactorisedConv` helper for k×k→(1×k + k×1) factorisation.
* `InceptionNet.build_from_config()` consumes a *dict* (typically the
  `model:` section of a YAML config) and returns a ready‑to‑train model.

This preserves the sensible defaults discussed previously, so you can
still instantiate the class directly without a config.
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from implementations.classifier import Classifier
from utils.config import ClassifierConfig, FactorisedConvConfig, InceptionStageConfig, ModelConfig, InceptionBlockConfig, StemConfig

# -----------------------------------------------------------------------------
# 1.  Low‑level utilities
# -----------------------------------------------------------------------------
class FactorisedConv(nn.Module):
    """Factorise a k×k convolution into (1×k) + (k×1) to save FLOPs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = False,
        use_bn: bool = True,
        bn_before_act: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        def bn_relu(c: int) -> nn.Sequential:
            return (
                nn.Sequential(nn.BatchNorm2d(c), nn.ReLU(inplace=True))
                if bn_before_act
                else nn.Sequential(nn.ReLU(inplace=True), nn.BatchNorm2d(c))
            )

        if kernel_size == 1:
            layers = [nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=bias)]
            if use_bn:
                layers += [bn_relu(out_channels)]
            else:
                layers += [nn.ReLU(inplace=True)]
        else:
            layers = [
                nn.Conv2d(in_channels, out_channels, (1, kernel_size), stride=stride, padding=(0, padding), bias=bias)
            ]
            if use_bn:
                layers += [bn_relu(out_channels)]
            else:
                layers += [nn.ReLU(inplace=True)]

            layers += [
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(padding, 0), bias=bias)
            ]
            if use_bn:
                layers += [bn_relu(out_channels)]
            else:
                layers += [nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # noqa: N802
        return self.block(x)
    
    @classmethod
    def from_config(
        cls,
        cfg: Union[FactorisedConvConfig, dict]
    ) -> FactorisedConv:
        """
        Construct a FactorisedConv from a config object or raw dict.
        """
        if isinstance(cfg, dict):
            cfg = FactorisedConvConfig(**cfg)

        return cls(
            in_channels   = cfg.in_channels,
            out_channels  = cfg.out_channels,
            kernel_size   = cfg.kernel_size,
            stride        = cfg.stride,
            padding       = cfg.padding,
            bias          = cfg.bias,
            use_bn        = cfg.use_bn,
            bn_before_act = cfg.bn_before_act,
        )


class SE(nn.Module):
    """Squeeze‑and‑Excitation."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: N802
        b, c, *_ = x.shape
        y = self.avg(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# -----------------------------------------------------------------------------
# 2.  Inception block
# -----------------------------------------------------------------------------
class InceptionBlock(nn.Module):
    """Inception block with four branches + optional SE and residual."""

    def __init__(self, in_channels: int, cfg: InceptionBlockConfig):
        super().__init__()
        out_channels = cfg.b0 + cfg.b1[1] + cfg.b2[1] + cfg.pool_proj
        self.residual = cfg.residual and out_channels == in_channels

        # Branch 0 — 1×1
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, cfg.b0, 1, bias=False),
            nn.BatchNorm2d(cfg.b0) if cfg.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # Branch 1 — 1×1 → 3×3
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, cfg.b1[0], 1, bias=False),
            nn.BatchNorm2d(cfg.b1[0]) if cfg.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            FactorisedConv.from_config(
                {
                    "in_channels": cfg.b1[0],
                    "out_channels": cfg.b1[1],
                    "kernel_size": 3,
                    "use_bn": cfg.use_bn,
                    "bn_before_act": cfg.bn_before_act,
                }
            ),
        )

        # Branch 2 — 1×1 → 3×3 → 3×3 (≈5×5)
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, cfg.b2[0], 1, bias=False),
            nn.BatchNorm2d(cfg.b2[0]) if cfg.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            FactorisedConv.from_config(
                {
                    "in_channels": cfg.b2[0],
                    "out_channels": cfg.b2[1],
                    "kernel_size": 3,
                    "use_bn": cfg.use_bn,
                    "bn_before_act": cfg.bn_before_act,
                }
            ),
            FactorisedConv.from_config(
                {
                    "in_channels": cfg.b2[1],
                    "out_channels": cfg.b2[1],
                    "kernel_size": 3,
                    "use_bn": cfg.use_bn,
                    "bn_before_act": cfg.bn_before_act,
                }
            ),
        )

        # Branch 3 — 3×3 MaxPool → 1×1 conv
        self.b3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, cfg.pool_proj, 1, bias=False),
            nn.BatchNorm2d(cfg.pool_proj) if cfg.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        self.use_se = cfg.use_se
        if cfg.use_se:
            self.se = SE(out_channels, reduction=cfg.se_reduction)

        self.dropout = nn.Dropout2d(cfg.dropout_p) if cfg.dropout_p > 0 else nn.Identity()

    # ------------------------------------------------------------------
    # factory
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, in_channels: int, cfg: InceptionBlockConfig | dict) -> "InceptionBlock":
        if not isinstance(cfg, InceptionBlockConfig):
            cfg = InceptionBlockConfig(**cfg)  # type: ignore[arg-type]
        return cls(in_channels, cfg)

    def forward(self, x: Tensor) -> Tensor:  # noqa: N802
        out = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], dim=1)
        if self.use_se:
            out = self.se(out)
        out = self.dropout(out)
        if self.residual:
            out = out + x
        return out



# -----------------------------------------------------------------------------
# 3.  Network stem + stage wrapper
# -----------------------------------------------------------------------------
class Stem(nn.Module):
    """
    The network stem, now fully configurable via StemConfig.
    """
    def __init__(self, cfg: StemConfig):
        super().__init__()
        conv_pad = cfg.conv_padding if cfg.conv_padding is not None else cfg.kernel_size // 2
        pool_pad = cfg.pool_padding if cfg.pool_padding is not None else cfg.pool_kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv2d(
                cfg.in_channels,
                cfg.init_channels,
                cfg.kernel_size,
                stride=cfg.conv_stride,
                padding=conv_pad,
                bias=cfg.bias,
            ),
            nn.BatchNorm2d(cfg.init_channels) if cfg.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(cfg.pool_kernel_size, cfg.pool_stride, padding=pool_pad),
        )

    @classmethod
    def from_config(cls, cfg: StemConfig) -> Stem:
        return cls(cfg)

    def forward(self, x: Tensor) -> Tensor:
        """
        Pass input through the stem conv → BN → ReLU → pool sequence.
        """
        return self.stem(x)


class InceptionStage(nn.Module):
    """Sequence of Inception blocks with optional down‑sampling."""

    def __init__(self, in_channels: int, cfg: InceptionStageConfig):
        super().__init__()
        blocks: list[nn.Module] = []
        c = in_channels
        for block_cfg in cfg.blocks:
            blocks.append(InceptionBlock.from_config(c, block_cfg))
            c = block_cfg.b0 + block_cfg.b1[1] + block_cfg.b2[1] + block_cfg.pool_proj
        if cfg.downsample:
            blocks.append(nn.MaxPool2d(3, stride=2, padding=1))
        self.stage = nn.Sequential(*blocks)
        self.out_channels = c

    @classmethod
    def from_config(cls, in_channels: int, cfg: InceptionStageConfig | dict) -> "InceptionStage":
        if not isinstance(cfg, InceptionStageConfig):
            cfg = InceptionStageConfig(**cfg)  # type: ignore[arg-type]
        return cls(in_channels, cfg)

    def forward(self, x: Tensor) -> Tensor:  # noqa: N802
        return self.stage(x)


# -----------------------------------------------------------------------------
# 4.  Top‑level network
# -----------------------------------------------------------------------------
class InceptionNet(nn.Module):
    def __init__(self, num_classes: int, cfg: ModelConfig):
        super().__init__()
        self.stem = Stem.from_config(cfg.stem_cfg)

        stages: list[InceptionStage] = []
        c = cfg.stem_cfg.init_channels
        for stage_cfg in cfg.stages:
            stage = InceptionStage.from_config(c, stage_cfg)
            stages.append(stage)
            c = stage.out_channels
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = Classifier.from_config(c, num_classes, cfg.classifier)

        self.apply(self._kaiming_init)

    # ----------------------------- helpers ------------------------------
    @staticmethod
    def _kaiming_init(m: nn.Module) -> None:
        """
        Kaiming initialization for Conv2d, BatchNorm2d, Linear.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    @classmethod
    def build_from_config(
        cls,
        model_cfg: Union[ModelConfig, dict],
        num_classes: int
    ) -> "InceptionNet":
        if isinstance(model_cfg, dict):
            model_cfg = ModelConfig(**model_cfg)
        return InceptionNet(
            num_classes=num_classes,
            cfg=model_cfg,
        )
