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

from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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
# 2. Inception block
# -----------------------------------------------------------------------------
class InceptionBlock(nn.Module):
    """Improved Inception block with four branches and optional SE / residual."""

    def __init__(
        self,
        in_channels: int,
        *,
        b0: int,
        b1: Tuple[int, int],  # (reduce, out)
        b2: Tuple[int, int],
        pool_proj: int,
        use_bn: bool = True,
        use_se: bool = False,
        se_reduction: int = 16,
        residual: bool = False,
        dropout_p: float = 0.0,
        bn_before_act: bool = True,
    ) -> None:
        super().__init__()
        out_channels = b0 + b1[1] + b2[1] + pool_proj
        self.residual = residual and out_channels == in_channels

        # Branch 0 — 1×1
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, b0, 1, bias=False),
            nn.BatchNorm2d(b0) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # Branch 1 — 1×1 → 3×3
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, b1[0], 1, bias=False),
            nn.BatchNorm2d(b1[0]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            FactorisedConv(b1[0], b1[1], 3, use_bn=use_bn, bn_before_act=bn_before_act),
        )

        # Branch 2 — 1×1 → 3×3 → 3×3 (≈5×5)
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, b2[0], 1, bias=False),
            nn.BatchNorm2d(b2[0]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            FactorisedConv(b2[0], b2[1], 3, use_bn=use_bn, bn_before_act=bn_before_act),
            FactorisedConv(b2[1], b2[1], 3, use_bn=use_bn, bn_before_act=bn_before_act),
        )

        # Branch 3 — 3×3 MaxPool → 1×1 conv
        self.b3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, 1, bias=False),
            nn.BatchNorm2d(pool_proj) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        self.use_se = use_se
        if use_se:
            self.se = SE(out_channels, reduction=se_reduction)

        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

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
    def __init__(self, *, in_channels: int = 3, out_channels: int = 64, use_bn: bool = True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: N802
        return self.stem(x)


class InceptionStage(nn.Module):
    def __init__(self, in_channels: int, block_cfgs: List[Dict[str, Any]], downsample: bool):
        super().__init__()
        blocks = []
        c = in_channels
        for cfg in block_cfgs:
            blocks.append(InceptionBlock(c, **cfg))
            c = cfg["b0"] + cfg["b1"][1] + cfg["b2"][1] + cfg["pool_proj"]
        if downsample:
            blocks.append(nn.MaxPool2d(3, stride=2, padding=1))
        self.stage = nn.Sequential(*blocks)
        self.out_channels = c

    def forward(self, x: Tensor) -> Tensor:  # noqa: N802
        return self.stage(x)


# -----------------------------------------------------------------------------
# 4.  InceptionNet top‑level
# -----------------------------------------------------------------------------
class InceptionNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        stage_cfgs: List[List[Dict[str, Any]]],
        init_channels: int = 64,
        use_bn: bool = True,
        dropout_cls: float = 0.5,
    ) -> None:
        super().__init__()
        self.stem = Stem(out_channels=init_channels, use_bn=use_bn)

        stages: List[InceptionStage] = []
        c = init_channels
        for idx, block_cfgs in enumerate(stage_cfgs):
            stage = InceptionStage(c, block_cfgs, downsample=idx < len(stage_cfgs) - 1)
            stages.append(stage)
            c = stage.out_channels
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(dropout_cls), nn.Linear(c, num_classes))

        self._init()

    # --------------------------------------------------------------
    # Factory from YAML/dict ----------------------------------------------------
    # --------------------------------------------------------------
    @staticmethod
    def build_from_config(cfg: Dict[str, Any], num_classes: int) -> "InceptionNet":
        return InceptionNet(
            num_classes=num_classes,
            stage_cfgs=cfg["stage_cfgs"],
            init_channels=cfg.get("init_channels", 64),
            use_bn=cfg.get("use_bn", True),
            dropout_cls=cfg.get("dropout_cls", 0.5),
        )

    # --------------------------------------------------------------
    # Private helpers -----------------------------------------------------------
    # --------------------------------------------------------------
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:  # noqa: N802
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)