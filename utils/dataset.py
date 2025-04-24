import os
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# -----------------------------------------------------------------------------
# 1. Data loading helpers
# -----------------------------------------------------------------------------
cwd = os.getcwd()
# Select current working directory if possible, otherwise use default
if cwd:
    INPUT_PATH = Path(cwd)
else:
    INPUT_PATH = Path("/gpfs/home/nct/nct01117/DL1/")  # adjust if needed


def load_mame(dataframe: bool = False):
    """Return train/val/test splits for the MAMe dataset."""

    dataset = pd.read_csv(INPUT_PATH / "mame-dataset" / "MAMe_dataset.csv")
    splits = {}
    for split in ["train", "val", "test"]:
        df = dataset[dataset["Subset"] == split]
        paths = [INPUT_PATH / "mame-dataset" / "data" / fp for fp in df["Image file"].tolist()]
        splits[split] = (np.array(paths), np.array(df["Medium"].tolist()))
    if dataframe:
        out = {}
        for split, (paths, labels) in splits.items():
            out[split] = pd.DataFrame({"filename": paths, "class": labels})
        return out["train"], out["val"], out["test"]
    return splits["train"], splits["val"], splits["test"]


# -----------------------------------------------------------------------------
# 2. Torch dataset wrapper
# -----------------------------------------------------------------------------
class MAMeDataset(Dataset):
    def __init__(self, dataframe, transform=None, classes=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.classes = sorted(self.df["class"].unique()) if classes is None else classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["filename"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[row["class"]]


# -----------------------------------------------------------------------------
# 3. DataLoader creation with optional augmentations
# -----------------------------------------------------------------------------
def create_data_loaders(cfg: dict, use_cfg: bool = True):
    """
    Create and return the train, validation, and test DataLoader instances based on the provided configuration,
    with optional augmentation toggled via config.

    Args:
        cfg (dict): Configuration dictionary containing dataset, augmentation, and training parameters.
        use_cfg (bool): Whether to use cfg values (e.g., image size, batch size, augmentation). Defaults to True.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    df_train, df_val, df_test = load_mame(dataframe=True)
    classes = sorted(df_train["class"].unique())

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    default_size = 256
    img_size = cfg.get("dataset", {}).get("img_size", default_size) if use_cfg else default_size

    default_bs = 64
    bs = cfg.get("training", {}).get("batch_size", default_bs) if use_cfg else default_bs

    aug_cfg = cfg.get("augmentations", {}) if use_cfg else {}
    active_aug = aug_cfg.get("active", False)

    normalize_cfg = aug_cfg.get("normalize", {})
    mean = normalize_cfg.get("mean", [0.485, 0.456, 0.406])
    std = normalize_cfg.get("std", [0.229, 0.224, 0.225])

    resize_dim = tuple(aug_cfg.get("resize", [img_size, img_size]))

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    base_transform = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    if active_aug:
        transform_train = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomHorizontalFlip() if aug_cfg.get("random_horizontal_flip", False) else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(aug_cfg.get("random_rotation", 0)),
            transforms.RandomResizedCrop(resize_dim, scale=tuple(aug_cfg.get("scale", [0.8, 1.0]))),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        transform_val = base_transform
    else:
        transform_train = base_transform
        transform_val = base_transform

    # ------------------------------------------------------------------
    # Datasets & DataLoaders
    # ------------------------------------------------------------------
    train_ds = MAMeDataset(df_train, transform_train, classes)
    val_ds = MAMeDataset(df_val, transform_val, classes)
    test_ds = MAMeDataset(df_test, transform_val, classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    return (train_loader, val_loader, test_loader), classes
