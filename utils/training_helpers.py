import importlib
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import Config
import torch.optim as optim


# --------------------------------------------------------------------------- #
#                                BUILDERS                                     #
# --------------------------------------------------------------------------- #

def _get_model(cfg: Config, num_classes: int) -> torch.nn.Module:
    name = cfg.model.type
    module_map = {
        "InceptionNet":         "implementations.inception",
        "StandardCNN":          "implementations.standard",
        "TransferLearningCNN":  "implementations.transfer_learning",
    }
    if name not in module_map:
        raise ValueError(f"Unknown model '{name}'")

    mod = importlib.import_module(module_map[name])
    
    ModelCls = getattr(mod, name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return ModelCls.build_from_config(
        cfg.model.model_dump(), num_classes
    ).to(device)


def _get_optimizer(
    cfg: Config, model: torch.nn.Module
) -> torch.optim.Optimizer:
    opt_name = cfg.training.optimizer.type.lower()
    params   = cfg.training.optimizer.params or {}

    params.update(params)                       # YAML overrides params

    if opt_name == "sgd":
        return optim.SGD(model.parameters(), **params)
    elif opt_name == "adam":
        return optim.Adam(model.parameters(), **params)
    elif opt_name == "adamw":
        return optim.AdamW(model.parameters(), **params)
    elif opt_name == "rmsprop":
        return optim.RMSprop(model.parameters(), **params)
    else:
        raise ValueError(f"Unsupported optimiser '{opt_name}'")


def _get_scheduler(
    cfg: Config, optimizer: torch.optim.Optimizer
) -> optim.lr_scheduler._LRScheduler | None:
    sch_cfg = cfg.training.scheduler
    if sch_cfg.type is None or sch_cfg.type.lower() in {"none", ""}:
        return None

    name   = sch_cfg.type.lower()
    params = sch_cfg.params or {}

    if name == "cosine":
        # sensible default: T_max = #epochs
        params.setdefault("T_max", cfg.training.epochs)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    elif name == "step":
        params.setdefault("step_size", 30)
        params.setdefault("gamma", 0.1)
        return optim.lr_scheduler.StepLR(optimizer, **params)
    elif name == "plateau":
        params.setdefault("mode", "max")
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    else:
        raise ValueError(f"Unsupported scheduler '{name}'")


def train_one_epoch(model, loader: DataLoader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    tot = 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * imgs.size(0)
        running_correct += preds.eq(labels).sum().item()
        tot += imgs.size(0)

    return running_loss / tot, running_correct / tot


def evaluate(model, loader: DataLoader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    tot = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val", leave=False):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            loss_sum += loss.item() * imgs.size(0)
            correct += preds.eq(labels).sum().item()
            tot += imgs.size(0)

    return loss_sum / tot, correct / tot
