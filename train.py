import argparse
from pathlib import Path
import sys

import torch
import torch.optim as optim
from utils.config import Config
from utils.dataset import create_data_loaders
from utils.training_helpers import (
    train_one_epoch, evaluate, _get_model, _get_optimizer, _get_scheduler
)
from utils.hyperparameter_search import range_division_grid_search


def build_from_cfg(
    cfg: Config, num_classes: int, device: torch.device
) -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    optim.lr_scheduler._LRScheduler | None,
    torch.nn.Module,
    torch.amp.GradScaler,
]:
    """Single call that gives you every component you need to start training."""
    model = _get_model(cfg, num_classes).to(device)
    optimizer = _get_optimizer(cfg, model)
    scheduler = _get_scheduler(cfg, optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device)
    return model, optimizer, scheduler, criterion, scaler


def main() -> None:
    parser = argparse.ArgumentParser("Train any model on MAMe (or your own)")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to YAML config (omit for pure defaults)",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run hyperparameter search before full training",
    )
    args = parser.parse_args()

    # Load config
    cfg = Config.load(args.config)

    # Prepare data
    (train_loader, val_loader, test_loader), classes = create_data_loaders(cfg)
    num_classes = len(classes)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optional hyperparameter tuning
    if args.tune and cfg.get("tuning.param_grid", None):
        print("Starting hyperparameter search...")
        best_params, best_score = range_division_grid_search(
            cfg,
            param_grid=cfg.get("tuning.param_grid"),
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            device=device,
            relative_epsilon=cfg.get("tuning.relative_epsilon", 0.05),
            max_depth=cfg.get("tuning.max_depth", 5),
            num_candidates=cfg.get("tuning.num_candidates", 3),
            search_epochs=cfg.get("tuning.search_epochs", 3),
            search_csv=cfg.get("tuning.search_csv", None),
            silent=cfg.get("tuning.silent", False)
        )
        print(f"Best hyperparameters: {best_params}")
        print(f"Best validation score: {best_score:.4f}")

        # dump tuned config
        tuned_name = args.config.stem + ".tuned.yaml"
        tuned_path = args.config.with_name(tuned_name)
        cfg.dump_yaml(tuned_path)
        print(f"Tuned config written to: {tuned_path}")

    # Build model, optimizer, scheduler, criterion, scaler
    model, optimizer, scheduler, criterion, scaler = build_from_cfg(
        cfg, num_classes, device
    )
    print(f"Model: {model}")   
    # State file
    state_file = Path(f"{cfg.model.training_mode}_{cfg.model.type}_mame.pth")
    if state_file.exists():
        print(f"Overwriting model state from {state_file}")

    # Early stopping setup
    patience = cfg.training.get("patience", 20)
    print(f"Patience for early stopping: {patience} epochs")
    print(f"Training for {cfg.training.epochs} epochs")
    epochs_no_improve = 0
    best_acc = 0.0

    # Training loop
    for epoch in range(1, cfg.training.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            # ReduceLROnPlateau expects the metric
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        print(
            f"Epoch {epoch:03d}/{cfg.training.epochs}: "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
        )

        # Save best model and reset patience counter
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), state_file)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs with no improvement over {epochs_no_improve}.")
            break

    # Final test
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(state_file))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.3f}")


if __name__ == "__main__":
    main()