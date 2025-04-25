# Flexible CNN Training Framework

Train and tune modern convolutional neural networks (CNN) with **one command and one YAML file**. This repository wraps a clean PyTorch training loop, multiple reference architectures, a reproducible hyper‑parameter search, a ready‑to‑use loader for the [MAMe fine‑art dataset](https://mame-datasets.github.io/), and utilities for evaluation and one‑off scripts.

## ✨ Key features
- **Plug‑and‑play architectures** – *StandardCNN*, *InceptionNet*, *InceptionNetV3* and *TransferLearningCNN* (VGG‑19 backbone) selectable via `model.type`.
- **Single‑file configuration** – every experiment described in a human‑readable YAML; no code changes needed.
- **Recursive grid search** – coordinate‑wise search that quickly narrows numeric, boolean **and** categorical hyper‑parameters.
- **Mixed‑precision & GPU ready** – automatic casting and gradient scaling out of the box.
- **Evaluation & reporting** – run `test.py` to compute accuracy, precision, recall, F1, confusion matrix, and export as text, CSV & LaTeX tables.
- **Extensible** – drop your own model under `implementations/` and add one import in `utils/training_helpers.py`.

## ⚒ Installation
```bash
# 1. create env (Python ≥3.9)
conda create -n cnn-env python=3.10
conda activate cnn-env

# 2. install deps
pip install -r requirements.txt
```

> **Note**  
> GPU training requires a CUDA‑enabled version of **PyTorch**.

## 🚀 Quick start

### 1. Prepare the dataset
Download **MAMe** (≌12 GB) and unpack it so it looks like:
```
mame-dataset/
├── data/                 # images
│   └── 00000.jpg
├── MAMe_dataset.csv      # metadata & splits
```
The loader looks for this directory in the current working directory. For a different dataset, provide a CSV with `filename,class` columns and update `utils/dataset.py`.

### 2. Train a model
```bash
time python train.py --config configs/standardSearch.yaml
```
This will:
1. Build the model from the YAML  
2. Train for `training.epochs` epochs  
3. Save the best checkpoint as `best_<model>.pth`

### 3. Hyper‑parameter search (optional)
```bash
time python train.py --config configs/standardSearch.yaml --tune
```
Range‑division grid search recurses over every entry in `tuning.param_grid` until convergence. Results are logged to CSV (`tuning.search_csv`) and the best set is written back into the live config object before full training.

### 4. Evaluate a trained model
```bash
python test.py --config configs/standardSearch.yaml --weights best_StandardCNN.pth --output results/StandardCNN
```
Generates:
- `metrics.txt`: overall & per-class metrics  
- `confusion_matrix.csv`  
- LaTeX tables: `metrics.tex` & `confusion_matrix.tex`

## 🔧 Utility scripts
Additional one-off utilities live in the `scripts/` folder. For example:
```
scripts/
├── preprocess.py           # custom data transforms
├── download_weights.py     # download pretrained backbones
└── ...
```
Modify or extend as needed.

## 🔧 Configuration reference
Each section of `config.yaml` is described in code comments and Pydantic docstrings (`utils/config.py`). See `configs/standardSearch.yaml` for a full example.

## 📁 Repository layout
```
.
├── train.py                   # Training entry point
├── test.py                    # Evaluation script
├── implementations/           # Model definitions
│   ├── classifier.py
│   ├── standard.py
│   ├── inception.py
│   ├── inceptionv3.py
│   └── transfer_learning.py
├── utils/
│   ├── config.py
│   ├── dataset.py
│   ├── training_helpers.py
│   └── hyperparameter_search.py
├── scripts/                   # Utility and one-off scripts
│   └── *.py
├── configs/                   # YAML experiment configs
│   └── *.yaml
├── requirements.txt
└── README.md
```

## 📜 License
MIT © 2025 Antonio Lobo‑Santos & Alejandro Guzmán‑Requena

