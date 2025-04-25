from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml
from pydantic import BaseModel, Field


class DictLikeModel(BaseModel):
    def get(self, path: str, default: Any = None) -> Any:
        parts = path.split(".")
        cur: Any = self
        for p in parts:
            if isinstance(cur, BaseModel):
                cur = getattr(cur, p, default)
            elif isinstance(cur, dict):
                cur = cur.get(p, default)
            else:
                return default
        return cur

class FactorisedConvConfig(DictLikeModel):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: Optional[int] = None
    bias: bool = False
    use_bn: bool = True
    bn_before_act: bool = True


class ConvBlockConfig(DictLikeModel):
    num_blocks: int
    filters: int
    kernel_size: int
    use_bn: bool
    pool_kernel_size: int
    pool_stride: int

class InceptionBlockConfig(DictLikeModel):
    b0: int
    b1: List[int]
    b2: List[int]
    pool_proj: int
    use_bn: bool = True
    use_se: bool = False
    residual: bool = False
    se_reduction: int = 16
    dropout_p: float = 0.0
    bn_before_act: bool = True



class StageCfgV3(DictLikeModel):
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
    layers: List[int] = Field(default_factory=lambda: [4096, 4096])
    dropout: float = 0.5
    activation: str = "ReLU"

class ClassifyingModel(DictLikeModel):
    classifier: ClassifierConfig = ClassifierConfig()

class StandardCNNConfig(ClassifyingModel):
    avg_pool2d: bool = True

    conv_blocks: Optional[List[ConvBlockConfig]] = Field(
        default=None,
        description="List of conv-blocks for StandardCNN"
    )

class StemConfig(DictLikeModel):
    """
    Configuration for the network stem (initial Conv2d + BN + ReLU + Pool).
    Defaults match the original constants: 7×7 conv with stride 2, padding 3, followed by 3×3 pool.
    """
    in_channels: int = 3
    init_channels: int = 64
    kernel_size: int = 7
    conv_stride: int = 2
    conv_padding: Optional[int] = None  # defaults to kernel_size // 2 if None
    bias: bool = False
    use_bn: bool = True
    pool_kernel_size: int = 3
    pool_stride: int = 2
    pool_padding: Optional[int] = None  # defaults to pool_kernel_size // 2 if None

class InceptionStageConfig(BaseModel):
    blocks: List[InceptionBlockConfig] = []
    downsample: bool = True

class InceptionConfig(ClassifyingModel):
    stages: List[InceptionStageConfig]  = []
    stem_cfg: StemConfig = StemConfig()

class TransferLearningConfig(ClassifyingModel):
    trainable_layers: int = 0
    weights_path: str = "vgg19.pth"

# How to make inheritance exclusive (all classes are parents but only one can be the parent
# of an instance)
class ModelConfig(StandardCNNConfig, TransferLearningConfig, InceptionConfig):
    type: str = "InceptionNet"
    training_mode: str = "train"

class DatasetConfig(DictLikeModel):
    name: str = "MAMe"
    img_size: int = 256


class OptimizerConfig(DictLikeModel):
    type: str = "sgd"
    params: Dict[str, Any] = Field(default_factory=dict)


class SchedulerConfig(DictLikeModel):
    type: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(DictLikeModel):
    epochs: int = 90
    batch_size: int = 64
    patience: int = 20
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()


class TuningConfig(DictLikeModel):
    """
    Hyperparameter tuning settings.
    """
    param_grid: Dict[str, List[Union[int, float]]] = Field(
        default_factory=dict,
        description="Ranges for parameters to tune, e.g. {'training.lr': [0.001, 0.01]}"
    )
    relative_epsilon: float = Field(
        0.05, description="Fraction of range width for convergence stopping"
    )
    max_depth: int = Field(5, description="Max recursion depth per parameter")
    num_candidates: int = Field(3, description="Number of points per sweep")
    search_epochs: int = Field(3, description="Epochs per evaluation during search")
    patience: int = Field(5, description="Patience for early stopping in full training")


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
        """
        Serialize this Config to a YAML file.

        Args:
            path: Path to write the YAML to.
        """
        # Convert the BaseModel (and all nested models) to plain dict
        data = self.model_dump()

        # Ensure parent directory exists
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Dump to YAML (no sorting of keys, for readability)
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)