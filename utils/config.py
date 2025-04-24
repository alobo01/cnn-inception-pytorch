from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml
from pydantic import BaseModel, Field


class DictLikeModel(BaseModel):
    def get(self, path: str, default: Any = None) -> Any:
        parts = path.split('.')
        cur: Any = self
        for p in parts:
            if isinstance(cur, BaseModel):
                cur = getattr(cur, p, default)
            elif isinstance(cur, dict):
                cur = cur.get(p, default)
            else:
                return default
        return cur


class ConvBlockConfig(DictLikeModel):
    num_blocks: int
    filters: int
    kernel_size: int
    use_bn: bool
    pool_kernel_size: int
    pool_stride: int


class StageCfg(DictLikeModel):
    b0: int
    b1: List[int]
    b2: List[int]
    pool_proj: int
    use_bn: bool = True
    use_se: bool = True
    residual: bool = True
    se_reduction: int = 16
    dropout_p: float = 0.0
    bn_before_act: bool = True
    use_dropblock: bool = False
    dropblock_size: int = 7
    dropblock_prob: float = 0.0
    drop_path_prob: float = 0.0


class ClassifierConfig(DictLikeModel):
    layers: List[int] = Field(default_factory=lambda: [2048, 1024])
    dropout: float = 0.6
    activation: str = "ReLU"


class ModelConfig(DictLikeModel):
    name: str = "best"
    type: str = "InceptionNet"

    init_channels: int = 64
    use_bn: bool = True
    input_size: List[int] = Field(default_factory=lambda: [256, 256])

    # For StandardCNN
    conv_blocks: Optional[List[ConvBlockConfig]] = Field(
        default=None,
        description="List of conv-blocks for StandardCNN"
    )

    stage_cfgs: List[List[StageCfg]] = Field(
        default_factory=lambda: [
            # Stage 0 — No regularization
            [StageCfg(b0=64,  b1=[48,64],  b2=[64,96],  pool_proj=32),
             StageCfg(b0=64,  b1=[48,64],  b2=[64,96],  pool_proj=32),
             StageCfg(b0=64,  b1=[48,64],  b2=[64,96],  pool_proj=32)],
            # Stage 1 — Light DropPath
            [StageCfg(b0=128, b1=[96,128], b2=[96,128], pool_proj=128, drop_path_prob=0.03),
             StageCfg(b0=128, b1=[96,128], b2=[96,128], pool_proj=128, drop_path_prob=0.05),
             StageCfg(b0=128, b1=[96,128], b2=[96,128], pool_proj=128, drop_path_prob=0.05),
             StageCfg(b0=128, b1=[96,128], b2=[96,128], pool_proj=128, drop_path_prob=0.05)],
            # Stage 2 — Moderate DropPath + DropBlock
            [StageCfg(b0=192, b1=[128,192], b2=[128,192], pool_proj=192,
                      drop_path_prob=0.08, use_dropblock=True, dropblock_prob=0.1, dropblock_size=7),
             StageCfg(b0=192, b1=[128,192], b2=[128,192], pool_proj=192,
                      drop_path_prob=0.10, use_dropblock=True, dropblock_prob=0.1, dropblock_size=7)],
        ]
    )
    using_multi_classifier: bool = True

    trainable_layers: int = 0
    weights_path: str = ""

    classifier: ClassifierConfig = ClassifierConfig()


class DatasetConfig(DictLikeModel):
    name: str = "default"
    img_size: int = 299


class OptimizerConfig(DictLikeModel):
    type: str = "rmsprop"
    params: Dict[str, Any] = Field(default_factory=lambda: {
        "lr": 0.001,
        "alpha": 0.99,
        "momentum": 0.9,
        "weight_decay": 1e-4
    })


class SchedulerConfig(DictLikeModel):
    type: Optional[str] = "cosine"
    params: Dict[str, Any] = Field(default_factory=lambda: {"T_max": 180})


class TrainingConfig(DictLikeModel):
    epochs: int = 180
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    patience: int = Field(5, description="Patience for early stopping in full training")
    label_smoothing: float = 0.1
    aux_weight: float = 0.3


class TuningConfig(DictLikeModel):
    param_grid: Dict[str, List[Union[int, float]]] = Field(
        default_factory=dict,
        description="Ranges for parameters to tune"
    )
    relative_epsilon: float = Field(
        0.05, description="Fraction of range width for convergence stopping"
    )
    max_depth: int = Field(5, description="Max recursion depth per parameter")
    num_candidates: int = Field(3, description="Number of points per sweep")
    search_epochs: int = Field(3, description="Epochs per evaluation during search")


class Config(DictLikeModel):
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    tuning: Optional[TuningConfig] = None

    @classmethod
    def load(cls, path: Union[str, Path] | None) -> Config:
        if path is None:
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def dump_yaml(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            yaml.safe_dump(self.dict(), f, sort_keys=False)
