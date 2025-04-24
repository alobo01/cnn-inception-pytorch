#!/usr/bin/env python
"""
saliency_from_model.py — Produce a 3×3 composite figure that contrasts the
saliency maps of three CNN architectures on the MAMe dataset, with labels
encoded numerically.

The figure layout is:
        ┌─────────┬────────────┬───────────┐
        │  Input  │  Saliency  │  Overlay  │  ← Standard CNN
        ├─────────┼────────────┼───────────┤
        │         │            │           │  ← InceptionNet
        ├─────────┼────────────┼───────────┤
        │         │            │           │  ← VGG19 Transfer
        └─────────┴────────────┴───────────┘

Example
-------
    python saliency_from_model.py \
        --models models/best_InceptionNet_mame.pth \
                 models/best_StandardCNN_mame.pth \
                 models/best_TransferLearningCNN_mame.pth \
        --outfile figs/saliency_grid.png

Dependencies: torch, torchvision.transforms, matplotlib, pandas.
"""

from __future__ import annotations
import sys
from pathlib import Path
import argparse
import importlib
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from PIL import Image

# Ensure implementations/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- Dynamic model loader registry ---
MODEL_REGISTRY = {
    "standard":  ("implementations.standard",          "StandardCNN"),
    "inception": ("implementations.inception",         "InceptionNet"),
    "transfer":  ("implementations.transfer_learning","TransferLearningCNN"),
}
LOADED_MODELS: dict[str, type] = {}
for key, (mod, cls) in MODEL_REGISTRY.items():
    module = importlib.import_module(mod)
    LOADED_MODELS[key] = getattr(module, cls)


def get_model(name: str, **kwargs) -> torch.nn.Module:
    try:
        return LOADED_MODELS[name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown model key: {name}")

plt.rcParams["figure.dpi"] = 300

# --- MAMe dataset loader with numeric label encoding ---
INPUT_PATH = Path(__file__).resolve().parent.parent

def load_mame() -> Tuple[List[Path], np.ndarray, List[Path], np.ndarray, List[Path], np.ndarray]:
    """Return train/val/test splits for MAMe, encoding string labels as integers."""
    df = pd.read_csv(INPUT_PATH / "mame-dataset" / "MAMe_dataset.csv")
    # Build a consistent mapping from literal Medium to integer codes
    unique_mediums = sorted(df["Medium"].unique())
    medium_to_idx = {medium: idx for idx, medium in enumerate(unique_mediums)}

    splits: dict[str, Tuple[List[Path], np.ndarray]] = {}
    for split in ["train", "val", "test"]:
        sub = df[df["Subset"] == split]
        paths = [INPUT_PATH / "mame-dataset" / "data" / p for p in sub["Image file"]]
        # Map each string label to its integer code
        labels = sub["Medium"].map(medium_to_idx).to_numpy()
        splits[split] = (paths, labels)

    return (*splits["train"], *splits["val"], *splits["test"]), medium_to_idx

# --- Model loading & saliency computation ---
def _load_model(pt_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    stem = pt_path.stem.lower()
    if "standard" in stem:
        model = get_model("standard", num_classes=num_classes)
    elif "inception" in stem:
        cfg = { ... }  # same as before
        model = LOADED_MODELS["inception"].build_from_config(cfg, num_classes=num_classes)
    elif "transfer" in stem:
        model = get_model("transfer", num_classes=num_classes, trainable_layers=0,
                          weights_path="vgg19.pth", classifier_layers=[128],
                          classifier_dropout=0.3, classifier_activation="relu",
                          device=device)
    else:
        raise ValueError(f"Cannot infer model type from: {pt_path.name}")
    state = torch.load(pt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


def _compute_saliency(model: torch.nn.Module, x: torch.Tensor) -> Tuple[np.ndarray, float, int]:
    x = x.unsqueeze(0).requires_grad_(True)
    out = model(x)
    prob, pred = torch.softmax(out, dim=1).max(1)
    out[0, pred].backward()
    sal = x.grad.abs().squeeze(0).max(0)[0]
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal.cpu().numpy(), prob.item(), pred.item()

# --- Main ---
def main():
    parser = argparse.ArgumentParser(
        description="Plot saliency for the single most confidently predicted image per model."
    )
    parser.add_argument(
        "--models", nargs=3, required=True,
        help="Paths to the three .pth files"
    )
    parser.add_argument(
        "--outfile", default="figs/saliency_grid.png",
        help="Where to save the 3×3 saliency grid"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="torch.device string (cpu or cuda)"
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    # Load MAMe splits and mapping
    (train_paths, train_labels,
     val_paths, val_labels,
     test_paths, test_labels), medium_to_idx = load_mame()
    paths, labels = test_paths, test_labels

    # Transform and prepare loader without ImageFolder
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    samples = [(transform(Image.open(p).convert("RGB")), int(l)) for p, l in zip(paths, labels)]
    loader = DataLoader(samples, batch_size=32, shuffle=False)
    num_classes = len(medium_to_idx)

    images: List[torch.Tensor] = []
    salmaps: List[np.ndarray] = []
    titles: List[str] = []

    for mp in args.models:
        pt = Path(mp)
        model = _load_model(pt, num_classes, device)
        best_prob, best_img = -1.0, None
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                probs = torch.softmax(model(imgs), dim=1)
                maxp, _ = probs.max(dim=1)
                idx = maxp.argmax().item()
                if maxp[idx] > best_prob:
                    best_prob = maxp[idx].item()
                    best_img = imgs[idx].cpu()
        sal, p, c = _compute_saliency(model, best_img.to(device))
        images.append(best_img)
        salmaps.append(sal)
        # Note: c is now the numeric label
        titles.append(f"{pt.stem} (p={p:.2f}, cls={c})")

    # Plot 3×3 grid
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for j, lab in enumerate(["Input", "Saliency", "Overlay"]):
        axes[0, j].set_title(lab)

    for i in range(3):
        img = images[i].permute(1, 2, 0).numpy()
        sal = salmaps[i]
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(sal, cmap="hot")
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(sal, cmap="hot", alpha=0.5)
        for j in range(3):
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(titles[i], rotation=90)

    plt.tight_layout()
    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    print(f"Saved → {out}")

if __name__ == "__main__":
    main()
