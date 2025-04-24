"""InceptionNet — fully YAML‑configurable implementation
=====================================================
This version exposes **every architectural knob** through a single
`build_from_config(cfg, num_classes)` factory so that the entire network
can be driven from a YAML file.
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from implementations.classifier import Classifier
from utils.config import ModelConfig, StageCfg

# -----------------------------------------------------------------------------  
# 0.  DropBlock & Stochastic Depth Helpers  
# -----------------------------------------------------------------------------
class DropBlock2d(nn.Module):
    """DropBlock: zeroes out contiguous regions in feature-maps."""
    def __init__(self, block_size: int, drop_prob: float):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        # compute gamma
        _, _, H, W = x.shape
        gamma = (
            self.drop_prob
            * (H * W)
            / ((H - self.block_size + 1) * (W - self.block_size + 1))
            / (self.block_size ** 2)
        )
        # sample mask
        mask = (torch.rand(x.shape[0], H, W, device=x.device) < gamma).float()
        block_mask = F.max_pool2d(
            mask.unsqueeze(1),
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2,
        )
        block_mask = 1 - block_mask.squeeze(1)
        # renormalize
        return x * block_mask.unsqueeze(1) * (
            block_mask.numel() / (block_mask.sum() + 1e-6)
        )


def drop_path(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    """Implements Stochastic Depth per-sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)
    return x.div(keep_prob) * binary_mask


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
            layers += [bn_relu(out_channels)] if use_bn else [nn.ReLU(inplace=True)]
        else:
            layers = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    (1, kernel_size),
                    stride=stride,
                    padding=(0, padding),
                    bias=bias,
                )
            ]
            layers += [bn_relu(out_channels)] if use_bn else [nn.ReLU(inplace=True)]
            layers += [
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(padding, 0), bias=bias)
            ]
            layers += [bn_relu(out_channels)] if use_bn else [nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        b, c, *_ = x.shape
        y = self.avg(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# -----------------------------------------------------------------------------  
# 2. Inception block  
# -----------------------------------------------------------------------------
class InceptionBlock(nn.Module):
    """Improved Inception block with four branches, optional SE, DropBlock & stochastic depth."""
    def __init__(
        self,
        in_channels: int,
        *,
        b0: int,
        b1: Tuple[int, int],
        b2: Tuple[int, int],
        pool_proj: int,
        use_bn: bool = True,
        use_se: bool = False,
        se_reduction: int = 16,
        residual: bool = False,
        dropout_p: float = 0.0,
        bn_before_act: bool = True,
        use_dropblock: bool = False,
        dropblock_size: int = 7,
        dropblock_prob: float = 0.1,
        drop_path_prob: float = 0.0,
    ) -> None:
        super().__init__()
        out_channels = b0 + b1[1] + b2[1] + pool_proj
        self.residual = residual and out_channels == in_channels
        self.drop_path_prob = drop_path_prob

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

        # Branch 2 — 1×1 → 3×3 → 3×3
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

        self.use_dropblock = use_dropblock
        if use_dropblock:
            self.dropblock = DropBlock2d(block_size=dropblock_size, drop_prob=dropblock_prob)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], dim=1)
        if self.use_se:
            out = self.se(out)
        out = self.dropout(out)
        if self.use_dropblock:
            out = self.dropblock(out)
        if self.residual:
            # stochastic depth / drop path
            out = drop_path(out, self.drop_path_prob, self.training) + x
        return out


# -----------------------------------------------------------------------------  
# 3.  Network stem + stage wrapper  
# -----------------------------------------------------------------------------
class Stem(nn.Module):
    """
    Inception-v3 style stem adapted for 256×256 input:
      3×3/2 → 3×3 → 3×3/2  →  max-pool/2
      1×1 → 3×3 → 1×1 → 3×3  (factorised) → avg-pool/2
    Output spatial size: 31×31 for 256×256 input.
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 32, use_bn: bool = True):
        super().__init__()
        act = nn.ReLU(inplace=True)
        bn  = lambda c: nn.BatchNorm2d(c) if use_bn else nn.Identity()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, bias=False), bn(32), act,       # 256→127
            nn.Conv2d(32, 32, 3, bias=False),             bn(32), act,       # 127→125
            nn.Conv2d(32, 64, 3, padding=1, bias=False),  bn(64), act,       # 125→125
            nn.MaxPool2d(3, stride=2, padding=1),                                # 125→63
            nn.Conv2d(64, 80, 1, bias=False),             bn(80), act,       # 63→63
            nn.Conv2d(80, 192, 3, bias=False),            bn(192), act,      # 63→61
            nn.MaxPool2d(3, stride=2, padding=1),                                # 61→31
        )
        self.out_ch = 192    # channel count after stem

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)

class InceptionBlock(nn.Module):
    """Improved Inception block driven by a StageCfg."""
    def __init__(self, in_channels: int, cfg: StageCfg) -> None:
        super().__init__()
        b0, b1, b2, pool_proj = cfg.b0, tuple(cfg.b1), tuple(cfg.b2), cfg.pool_proj
        use_bn, bn_before_act = cfg.use_bn, cfg.bn_before_act

        out_channels = b0 + b1[1] + b2[1] + pool_proj
        self.residual = cfg.residual and (out_channels == in_channels)
        self.drop_path_prob = cfg.drop_path_prob

        # Branch 0 — 1×1 conv
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, b0, 1, bias=False),
            nn.BatchNorm2d(b0) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # Branch 1 — 1×1 → factorised 3×3
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, b1[0], 1, bias=False),
            nn.BatchNorm2d(b1[0]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            FactorisedConv(b1[0], b1[1], 3, use_bn=use_bn, bn_before_act=bn_before_act),
        )

        # Branch 2 — 1×1 → 3×3 → 3×3
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, b2[0], 1, bias=False),
            nn.BatchNorm2d(b2[0]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            FactorisedConv(b2[0], b2[1], 3, use_bn=use_bn, bn_before_act=bn_before_act),
            FactorisedConv(b2[1], b2[1], 3, use_bn=use_bn, bn_before_act=bn_before_act),
        )

        # Branch 3 — max-pool → 1×1 conv
        self.b3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, 1, bias=False),
            nn.BatchNorm2d(pool_proj) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # Optional SE
        self.use_se = cfg.use_se
        if cfg.use_se:
            self.se = SE(out_channels, reduction=cfg.se_reduction)

        # Dropout & DropBlock
        self.dropout = nn.Dropout2d(cfg.dropout_p) if cfg.dropout_p > 0 else nn.Identity()
        self.use_dropblock = cfg.use_dropblock
        if cfg.use_dropblock:
            self.dropblock = DropBlock2d(block_size=cfg.dropblock_size,
                                         drop_prob=cfg.dropblock_prob)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], dim=1)
        if self.use_se:
            out = self.se(out)
        out = self.dropout(out)
        if self.use_dropblock:
            out = self.dropblock(out)
        if self.residual:
            out = drop_path(out, self.drop_path_prob, self.training) + x
        return out

class InceptionStage(nn.Module):
    """A full stage composed of multiple InceptionBlocks driven by StageCfgs."""
    def __init__(self, in_channels: int, block_cfgs: List[StageCfg], downsample: bool):
        super().__init__()
        blocks = []
        c = in_channels
        for cfg in block_cfgs:
            blocks.append(InceptionBlock(c, cfg))
            c = cfg.b0 + cfg.b1[1] + cfg.b2[1] + cfg.pool_proj
        if downsample:
            blocks.append(nn.MaxPool2d(3, stride=2, padding=1))

        self.stage = nn.Sequential(*blocks)
        self.out_channels = c

    def forward(self, x: Tensor) -> Tensor:
        return self.stage(x)

# ------------------------------------------------------------------  
# 4.  Auxiliary head helper  
# ------------------------------------------------------------------
class AuxHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(5),
            nn.Conv2d(in_ch, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 5, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(768, num_classes)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: Tensor) -> Tensor:           # noqa: D401
        x = self.net(x)
        return self.fc(torch.flatten(x, 1))


# ------------------------------------------------------------------  
# 5.  InceptionNet top-level (multi-classifier aware)  
# ------------------------------------------------------------------
class InceptionNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        stage_cfgs: list,
        use_bn: bool = True,
        classifier_cfg: ClassifierConfig,
        use_multi_classifier: bool = False,
    ) -> None:
        super().__init__()

        # ---------------- Stem + stages ----------------
        self.stem = Stem(use_bn=use_bn)
        c = self.stem.out_ch
        stages: list[InceptionStage] = []
        ch_after_stage: list[int] = []

        for idx, cfgs in enumerate(stage_cfgs):
            stage = InceptionStage(c, cfgs, downsample=(idx < len(stage_cfgs) - 1))
            stages.append(stage)
            c = stage.out_channels
            ch_after_stage.append(c)

        self.stages = nn.ModuleList(stages)
        self.pool   = nn.AdaptiveAvgPool2d(1)

        # ---------------- Auxiliary heads --------------
        self.using_multi_classifier = bool(use_multi_classifier)
        if self.using_multi_classifier:
            if len(ch_after_stage) < 3:
                raise ValueError("Need ≥3 stages to attach two aux-heads")
            self.aux1 = AuxHead(ch_after_stage[1], num_classes)
            self.aux2 = AuxHead(ch_after_stage[2], num_classes)

        # ---------------- Main classifier --------------
        # Shared classifier head
        self.classifier = Classifier.from_config(
            in_features=c,
            num_classes=num_classes,
            cfg=classifier_cfg,
        )
        self._init_weights()

    # ------------- factory -----------------
    @classmethod
    def build_from_config(
        cls,
        model_cfg: Union[ModelConfig, dict],
        num_classes: int,
    ) -> InceptionNet:
        if isinstance(model_cfg, dict):
            model_cfg = ModelConfig(**model_cfg)
        return cls(
            num_classes=num_classes,
            stage_cfgs=model_cfg.stage_cfgs,
            use_bn=model_cfg.use_bn,
            classifier_cfg=model_cfg.classifier,
            use_multi_classifier=model_cfg.using_multi_classifier,
        )

    # ------------- weight init -------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------- forward -----------------
    def forward(self, x: Tensor) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
        # 1) run the stem
        x0 = self.stem(x)         # Tensor after the stem

        # 2) run each stage in turn, saving the outputs
        x1 = self.stages[0](x0)   # output of stage 0
        x2 = self.stages[1](x1)   # output of stage 1
        x3 = self.stages[2](x2)   # output of stage 2
        # if you have more stages, continue feeding through them:
        for stage in self.stages[3:]:
            x3 = stage(x3)

        # 3) compute aux heads *only* on the saved tensors
        if self.using_multi_classifier and self.training:
            aux1 = self.aux1(x2)  # note: uses x2 (output of stage 1)
            aux2 = self.aux2(x3)  # uses x3 (output of stage 2)

        # 4) do global pooling + main classifier
        pooled = self.pool(x3)
        flat   = torch.flatten(pooled, 1)
        main   = self.classifier(flat)

        if self.using_multi_classifier and self.training:
            return main, aux1, aux2
        return main