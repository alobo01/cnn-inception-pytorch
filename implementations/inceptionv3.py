from __future__ import annotations
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from implementations.classifier import Classifier
from utils.config import (
    ModelConfig,
    StemConfig,
    InceptionStageConfig,
    InceptionBlockConfig,
    FactorisedConvConfig,
    ClassifierConfig,
)

# -----------------------------------------------------------------------------
# 0. DropBlock & Stochastic Depth Helpers
# -----------------------------------------------------------------------------
class DropBlock2d(nn.Module):
    def __init__(self, block_size: int, drop_prob: float):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        _, _, H, W = x.shape
        gamma = (
            self.drop_prob
            * (H * W)
            / ((H - self.block_size + 1) * (W - self.block_size + 1))
            / (self.block_size ** 2)
        )
        mask = (torch.rand(x.shape[0], H, W, device=x.device) < gamma).float()
        block_mask = F.max_pool2d(
            mask.unsqueeze(1),
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2,
        )
        block_mask = 1 - block_mask.squeeze(1)
        return x * block_mask.unsqueeze(1) * (
            block_mask.numel() / (block_mask.sum() + 1e-6)
        )


def drop_path(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)
    return x.div(keep_prob) * binary_mask


# -----------------------------------------------------------------------------
# 1. Low‑level utilities
# -----------------------------------------------------------------------------
class FactorisedConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int | None = None,
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
            layers: list[nn.Module] = [
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=bias)
            ]
            layers.append(bn_relu(out_channels) if use_bn else nn.ReLU(inplace=True))
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
            layers.append(bn_relu(out_channels) if use_bn else nn.ReLU(inplace=True))
            layers.append(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    (kernel_size, 1),
                    padding=(padding, 0),
                    bias=bias,
                )
            )
            layers.append(bn_relu(out_channels) if use_bn else nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    @classmethod
    def from_config(cls, cfg: Union[FactorisedConvConfig, dict]) -> FactorisedConv:
        if isinstance(cfg, dict):
            cfg = FactorisedConvConfig(**cfg)
        return cls(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            padding=cfg.padding,
            bias=cfg.bias,
            use_bn=cfg.use_bn,
            bn_before_act=cfg.bn_before_act,
        )


class SE(nn.Module):
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
    def __init__(self, in_channels: int, cfg: InceptionBlockConfig) -> None:
        super().__init__()
        out_ch = cfg.b0 + cfg.b1[1] + cfg.b2[1] + cfg.pool_proj
        self.residual = cfg.residual and out_ch == in_channels
        self.drop_path_prob = getattr(cfg, "drop_path_prob", 0.0)

        # Branch 0
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, cfg.b0, 1, bias=False),
            nn.BatchNorm2d(cfg.b0) if cfg.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        # Branch 1
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, cfg.b1[0], 1, bias=False),
            nn.BatchNorm2d(cfg.b1[0]) if cfg.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            FactorisedConv.from_config(
                {"in_channels": cfg.b1[0], "out_channels": cfg.b1[1], "kernel_size": 3,
                 "use_bn": cfg.use_bn, "bn_before_act": cfg.bn_before_act}
            ),
        )
        # Branch 2
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, cfg.b2[0], 1, bias=False),
            nn.BatchNorm2d(cfg.b2[0]) if cfg.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            FactorisedConv.from_config(
                {"in_channels": cfg.b2[0], "out_channels": cfg.b2[1], "kernel_size": 3,
                 "use_bn": cfg.use_bn, "bn_before_act": cfg.bn_before_act}
            ),
            FactorisedConv.from_config(
                {"in_channels": cfg.b2[1], "out_channels": cfg.b2[1], "kernel_size": 3,
                 "use_bn": cfg.use_bn, "bn_before_act": cfg.bn_before_act}
            ),
        )
        # Branch 3
        self.b3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, cfg.pool_proj, 1, bias=False),
            nn.BatchNorm2d(cfg.pool_proj) if cfg.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        self.use_se = cfg.use_se
        if cfg.use_se:
            self.se = SE(out_ch, reduction=cfg.se_reduction)

        self.dropout = nn.Dropout2d(cfg.dropout_p) if cfg.dropout_p > 0 else nn.Identity()
        self.use_dropblock = getattr(cfg, "use_dropblock", False)
        if self.use_dropblock:
            self.dropblock = DropBlock2d(cfg.dropblock_size, cfg.dropblock_prob)

    @classmethod
    def from_config(
        cls, in_channels: int, cfg: Union[InceptionBlockConfig, dict]
    ) -> InceptionBlock:
        if isinstance(cfg, dict):
            cfg = InceptionBlockConfig(**cfg)
        return cls(in_channels, cfg)

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


# -----------------------------------------------------------------------------
# 3. Stem & Stage
# -----------------------------------------------------------------------------
class Stem(nn.Module):
    def __init__(self, cfg: StemConfig):
        super().__init__()
        pad = cfg.conv_padding if cfg.conv_padding is not None else cfg.kernel_size // 2
        pool_pad = cfg.pool_padding if cfg.pool_padding is not None else cfg.pool_kernel_size // 2
        self.seq = nn.Sequential(
            nn.Conv2d(
                cfg.in_channels,
                cfg.init_channels,
                cfg.kernel_size,
                stride=cfg.conv_stride,
                padding=pad,
                bias=cfg.bias,
            ),
            nn.BatchNorm2d(cfg.init_channels) if cfg.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(cfg.pool_kernel_size, cfg.pool_stride, padding=pool_pad),
        )
        self.out_channels = cfg.init_channels

    @classmethod
    def from_config(cls, cfg: Union[StemConfig, dict]) -> Stem:
        if isinstance(cfg, dict):
            cfg = StemConfig(**cfg)
        return cls(cfg)

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class InceptionStage(nn.Module):
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
    def from_config(
        cls, in_channels: int, cfg: Union[InceptionStageConfig, dict]
    ) -> InceptionStage:
        if isinstance(cfg, dict):
            cfg = InceptionStageConfig(**cfg)
        return cls(in_channels, cfg)

    def forward(self, x: Tensor) -> Tensor:
        return self.stage(x)


# -----------------------------------------------------------------------------
# 4. Auxiliary head
# -----------------------------------------------------------------------------
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return self.fc(torch.flatten(x, 1))


# -----------------------------------------------------------------------------
# 5. Top-level network
# -----------------------------------------------------------------------------
class InceptionNetV3(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        # Stem
        self.stem = Stem.from_config(cfg.stem_cfg)
        c = self.stem.out_channels

        # Stages
        stages: list[nn.Module] = []
        for stage_cfg in cfg.stages:
            stage = InceptionStage.from_config(c, stage_cfg)
            stages.append(stage)
            c = stage.out_channels
        self.stages = nn.Sequential(*stages)

        # Pool & classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = Classifier.from_config(
            input_dim=c,
            num_classes=num_classes,
            cfg=cfg.classifier,
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
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

    @classmethod
    def build_from_config(
        cls,
        model_cfg: Union[ModelConfig, dict],
        num_classes: int,
    ) -> InceptionNetV3:
        if isinstance(model_cfg, dict):
            model_cfg = ModelConfig(**model_cfg)
        return cls(num_classes=num_classes, cfg=model_cfg)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        x = self.stem(x)
        # capture intermediate if needed for aux heads
        out = self.stages(x)
        pooled = self.pool(out)
        flat = torch.flatten(pooled, 1)
        main = self.classifier(flat)
        return main
