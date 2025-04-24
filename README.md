# Flexible CNN Training Framework

Train and tune modern convolutional neural networks (CNNs) with **one command and one YAML file**. This repository wraps a clean PyTorch training loop, three reference architectures, a reproducible hyper‑parameter search and a ready‑to‑use loader for the [MAMe fine‑art dataset](https://mame-datasets.github.io/).

## ✨ Key features
- **Plug‑and‑play architectures** – *StandardCNN*, *InceptionNet* and *TransferLearningCNN* (VGG‑19 backbone) selectable through `model.type`.
- **Single‑file configuration** – every experiment is described in a human‑readable YAML; no code changes needed.
- **Recursive grid search** – coordinate‑wise search that quickly narrows numeric, boolean **and** categorical hyper‑parameters.
- **Mixed‑precision & GPU ready** – automatic casting and gradient scaling out of the box.
- **Extensible** – drop your own model under `implementations/` and add one import in `utils/training_helpers.py`.

## ⚒ Installation
```bash
# 1. create env (Python ≥3.9)
conda create -n cnn-env python=3.10
conda activate cnn-env

# 2. install deps
pip install -r requirements.txt        # or pip install torch torchvision pydantic tqdm pandas pillow pyyaml
```

> **Note**  
> GPU training requires a CUDA‑enabled version of **PyTorch**.

## 🚀 Quick start

### 1. Prepare the dataset
Download **MAMe** (≌12 GB) and unpack it so the folder structure looks like:
```
mame-dataset/
├── data/                 # images
│   └── 00000.jpg
├── MAMe_dataset.csv      # metadata & splits
```
The loader looks for this directory in the current working directory.  
For a different dataset, provide your own CSV with `filename,class` columns and update `utils/dataset.py`.

### 2. Train a model
```bash
python train.py --config configs/standardSearch.yaml
```
The script will:
1. Build the model described in the YAML  
2. Train for `training.epochs` epochs  
3. Save the best checkpoint as `best_<model>.pth`  

### 3. Hyper‑parameter search (optional)
```bash
python train.py --config configs/standardSearch.yaml --tune
```
*Range‑division grid search* will recurse over every entry in `tuning.param_grid` until convergence.  
Results are logged to CSV (`tuning.search_csv`) and the best set is written back into the live config object before full training starts.

## 🔧 Configuration reference
A minimal file:

```yaml
dataset:
  img_size: 256          # input size for all transforms

model:
  type: StandardCNN      # ↳ choose: StandardCNN | InceptionNet | TransferLearningCNN

training:
  epochs: 90
  batch_size: 64
  lr: 0.1
```

See `configs/standardSearch.yaml` for a fully populated example including augmentations and tuning.

<details>
<summary>Model‑specific options</summary>

### StandardCNN
```yaml
model:
  type: StandardCNN
  init_channels: 64      # filters in first conv block
  blocks: [64, 128, 256] # override automatic doubling
  use_bn: true
  dropout_cls: 0.5
```

### TransferLearningCNN
```yaml
model:
  type: TransferLearningCNN
  weights_path: vgg19.pth    # download once with torchvision
  trainable_layers: 2        # unfreeze last N VGG blocks
  classifier:
    layers: [1024, 512]
    dropout: 0.5
    activation: relu
```

### InceptionNet
```yaml
model:
  type: InceptionNet
  init_channels: 64
  stage_cfgs:             # each list = one stage, each dict = one block
    - - {b0:64, b1:[48,64], b2:[64,96], pool_proj:32}
```
</details>

## 📁 Repository layout
```
.
├── train.py                      # CLI entry‑point
├── implementations/              # <‑‑ your models here
│   ├── standard.py               # StandardCNN
│   ├── inception.py              # InceptionNet
│   └── transfer_learning.py      # TransferLearningCNN
├── utils/
│   ├── config.py                 # pydantic‑backed config
│   ├── dataset.py                # CSV‑based DataLoader
│   ├── training_helpers.py       # builders & loop helpers
│   └── hyperparameter_search.py  # recursive grid search
└── configs/
    └── *.yaml                    # experiment configs
└── scripts/
    └── *.py                      # Single use scripts
```

## 📝 Citation
If you find this repo useful, please cite:
```text
@misc{cnnframework2025,
  title  = {Flexible CNN Training Framework},
  author = {Antonio Lobo-Santos and Alejandro Guzman-Requena},
  year   = {2025},
  url    = {https://github.com/guzmanalejandro/dl-mai}
}
```

## 📜 License
MIT © 2025 Antonio Lobo-Santos, Alejandro Guzmán-Requena

