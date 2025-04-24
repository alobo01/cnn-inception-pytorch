import importlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import Config
import torch.optim as optim


# --------------------------------------------------------------------------- #
#                                BUILDERS                                     #
# --------------------------------------------------------------------------- #


def _make_criterion(cfg: Config) -> nn.Module:
    return torch.nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)

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
        cfg.model.model_dump(),  # dumps into a dict
        num_classes
    ).to(device)

def _get_optimizer(cfg: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_name     = cfg.training.optimizer.type.lower()
    user_params  = dict(cfg.training.optimizer.params or {})

    # pull from nested params first, else fallback
    lr           = user_params.pop("lr", cfg.training.lr)
    weight_decay = user_params.pop("weight_decay", cfg.training.weight_decay)

    if opt_name == "sgd":
        momentum = user_params.pop("momentum", cfg.training.momentum)
        nesterov = user_params.pop("nesterov", False)
        return optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            **user_params,            # any extra flags
        )

    elif opt_name in {"adam", "adamw"}:
        # for Adam/AdamW, sensible defaults if neither top‐level nor nested
        betas = user_params.pop("betas", (0.9, 0.999))
        eps   = user_params.pop("eps", 1e-8)

        OptimCls = optim.AdamW if opt_name == "adamw" else optim.Adam
        return OptimCls(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            **user_params,
        )

    elif opt_name == "rmsprop":
        momentum = user_params.pop("momentum", cfg.training.momentum)
        eps      = user_params.pop("eps", 1e-8)
        return optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eps=eps,
            **user_params,
        )

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

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, aux_w=0.3):
    model.train()
    running_loss = running_correct = tot = 0
    multi = getattr(model, "using_multi_classifier", False)

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type):
            outs = model(imgs)
            if multi and isinstance(outs, tuple):
                main, aux1, aux2 = outs
                loss = (criterion(main, labels) +
                        aux_w * criterion(aux1, labels) +
                        aux_w * criterion(aux2, labels))
                preds = main.argmax(1)
            else:
                loss  = criterion(outs, labels)
                preds = outs.argmax(1)
    
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss   += loss.item() * imgs.size(0)
        running_correct += preds.eq(labels).sum().item()
        tot            += imgs.size(0)

    return running_loss / tot, running_correct / tot


def evaluate(model, loader, criterion, device, aux_w=0.3):
    model.eval()
    loss_sum = correct = tot = 0
    multi = getattr(model, "using_multi_classifier", False)

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val", leave=False):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
            outs = model(imgs)
            if multi and isinstance(outs, tuple):
                main, aux1, aux2 = outs
                loss = (criterion(main, labels) +
                        aux_w * criterion(aux1, labels) +
                        aux_w * criterion(aux2, labels))
                preds = main.argmax(1)
            else:
                loss  = criterion(outs, labels)
                preds = outs.argmax(1)

            loss_sum += loss.item() * imgs.size(0)
            correct  += preds.eq(labels).sum().item()
            tot      += imgs.size(0)

    return loss_sum / tot, correct / tot

