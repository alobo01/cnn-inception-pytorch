import csv
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, Tuple, List
from utils.config import Config
from utils.training_helpers import (
    _get_model, _get_optimizer,
    _get_scheduler, train_one_epoch, evaluate
)

# Global flag for silent mode
SILENT_MODE = False

def _get_attr(cfg: Config, path: str) -> Any:
    parts = path.split('.')
    obj = cfg
    for p in parts:
        if isinstance(obj, list) and (isinstance(p, int) or (isinstance(p, str) and p.isdigit())):
            p = int(p)
            obj = obj[p]
        else:
            obj = getattr(obj, p)
    return obj


def _set_attr(cfg: Config, path: str, value: Any):
    parts = path.split('.')
    obj = cfg
    for p in parts[:-1]:
        if isinstance(obj, list) and (isinstance(p, int) or (isinstance(p, str) and p.isdigit())):
            p = int(p)
            obj = obj[p]
        else:
            obj = getattr(obj, p)
    last = parts[-1]
    if isinstance(obj, list) and (isinstance(last, int) or (isinstance(last, str) and last.isdigit())):
        obj[int(last)] = value
    else:
        setattr(obj, last, value)


def train_and_evaluate(cfg: Config, train_loader, val_loader, num_classes: int,
                       device: torch.device, epochs: int) -> float:
    model = _get_model(cfg, num_classes).to(device)
    optimizer = _get_optimizer(cfg, model)
    scheduler = _get_scheduler(cfg, optimizer)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                _, val_acc = evaluate(model, val_loader, criterion, device)
                scheduler.step(val_acc)
            else:
                scheduler.step()

    _, val_acc = evaluate(model, val_loader, criterion, device)
    return val_acc


def _expand_wildcards(cfg: Config, param_grid: Dict[str, Any]) -> Dict[str, Any]:
    expanded: Dict[str, Any] = {}

    for path, spec in param_grid.items():
        if "[*]" not in path:
            expanded[path] = spec
            continue

        parts = path.split('.')

        def recurse(idx: int, obj: Any, prefix: List[str]):
            if idx >= len(parts):
                new_key = ".".join(prefix)
                expanded[new_key] = spec
                return

            part = parts[idx]
            if part == "[*]":
                if not isinstance(obj, (list, tuple)):
                    raise ValueError(f"Wildcard at '{'.'.join(prefix)}' but object is not list-like: {obj!r}")
                for i, elt in enumerate(obj):
                    recurse(idx + 1, elt, prefix + [str(i)])
            else:
                if isinstance(obj, dict) and part in obj:
                    next_obj = obj[part]
                elif hasattr(obj, part):
                    next_obj = getattr(obj, part)
                else:
                    try:
                        ix = int(part)
                        next_obj = obj[ix]
                    except Exception:
                        raise ValueError(f"Can't resolve '{part}' at '{'.'.join(prefix)}'")
                recurse(idx + 1, next_obj, prefix + [part])

        recurse(0, cfg, [])

    return expanded


def search_parameter(path: str,
                     candidate_values: List[Any],
                     current_best: Dict[str, Any],
                     cfg: Config,
                     train_loader,
                     val_loader,
                     num_classes: int,
                     device: torch.device,
                     epochs: int,
                     epsilon: float,
                     max_depth: int,
                     current_depth: int = 0,
                     narrowing_factor: int = 2
                     ) -> Tuple[Any, float, bool]:
    best_score = -np.inf
    best_value = current_best[path]
    original = _get_attr(cfg, path)

    # Determine type
    if isinstance(candidate_values, np.ndarray):
        kind = candidate_values.dtype.kind
        vals = candidate_values.tolist()
    else:
        vals = candidate_values
        if all(isinstance(v, bool) for v in vals):
            kind = 'b'
        elif all(isinstance(v, float) for v in vals):
            kind = 'n'
        elif all(isinstance(v, int) for v in vals):
            kind = 'i'
        else:
            kind = 'O'

    # Boolean or nominal
    if kind in ('b', 'O'):
        for cand in vals:
            _set_attr(cfg, path, cand)
            score = train_and_evaluate(cfg, train_loader, val_loader, num_classes, device, epochs)
            if not SILENT_MODE:
                print(f"[Depth {current_depth}] {path}={cand} -> {score:.4f}")
            if score > best_score:
                best_score = score
                best_value = cand
        _set_attr(cfg, path, original)
        return best_value, best_score, (kind == 'b')

    # Numeric search
    if kind == 'n':
        arr = np.array(vals, dtype=float)
    elif kind == 'i':
        arr = np.array(vals, dtype=int)
    else:
        raise ValueError(f"Unsupported type for {path}: {kind}")

    for cand in vals:
        _set_attr(cfg, path, cand)
        score = train_and_evaluate(cfg, train_loader, val_loader, num_classes, device, epochs)
        if not SILENT_MODE:
            print(f"[Depth {current_depth}] {path}={cand} -> {score:.4f}")
        if score > best_score:
            best_score = score
            best_value = cand
    _set_attr(cfg, path, original)

    width = arr.max() - arr.min()
    if width < epsilon or current_depth >= max_depth - 1:
        return best_value, best_score, width < epsilon

    low, high = arr.min(), arr.max()
    if np.isclose(best_value, low):
        new_low, new_high = low, low + width / narrowing_factor
    elif np.isclose(best_value, high):
        new_low, new_high = high - width / narrowing_factor, high
    else:
        half = width / narrowing_factor
        new_low, new_high = best_value - half / 2, best_value + half / 2

    new_vals = np.linspace(new_low, new_high, num=arr.shape[0])
    if kind == 'i':
        new_vals = np.array(new_vals, dtype=int)
        if new_vals[-2] == new_vals[-1] or new_vals[0] == new_vals[1]:
            return best_value, best_score, True

    return search_parameter(path, new_vals.tolist(), current_best, cfg,
                            train_loader, val_loader, num_classes, device,
                            epochs, epsilon, max_depth,
                            current_depth + 1, narrowing_factor)


def range_division_grid_search(
        cfg: Config,
        param_grid: Dict[str, Any],
        train_loader,
        val_loader,
        num_classes: int,
        device: torch.device,
        relative_epsilon: float = 0.05,
        max_depth: int = 5,
        num_candidates: int = 3,
        search_epochs: int = 3,
        search_csv: Path = None,
        silent: bool = False
    ) -> Tuple[Dict[str, Any], float]:
    global SILENT_MODE
    SILENT_MODE = silent

    if not SILENT_MODE:
        print("Starting hyperparameter search...")

    param_grid = _expand_wildcards(cfg, param_grid)
    csv_file = None
    writer = None
    if search_csv:
        csv_file = open(search_csv, 'w', newline='')
        writer = csv.writer(csv_file)
        header = list(param_grid.keys()) + ['depth', 'iteration', 'score']
        writer.writerow(header)

    best_params: Dict[str, Any] = {}
    orig_ranges: Dict[str, float] = {}
    current_ranges: Dict[str, Any] = {}
    converged: Dict[str, bool] = {}
    convergeable: List[str] = []

    for path, spec in param_grid.items():
        if isinstance(spec, list) and len(spec) == 2 and all(isinstance(x, (int, float, bool)) for x in spec):
            best_params[path] = spec[0]
            if isinstance(spec[0], (int, float)) and isinstance(spec[1], (int, float)):
                orig_ranges[path] = spec[1] - spec[0]
                param_grid[path] = (spec[0], spec[1])
                convergeable.append(path)
                spec = tuple(spec)
            else:
                orig_ranges[path] = None
            current_ranges[path] = spec
            converged[path] = False
        elif isinstance(spec, list):
            best_params[path] = _get_attr(cfg, path)
            orig_ranges[path] = None
            current_ranges[path] = spec
            converged[path] = False
        else:
            raise ValueError(f"Invalid spec for {path}: {spec}")

    overall_best = -np.inf
    iteration = 0

    while not all(converged[p] for p in convergeable):
        iteration += 1
        if not SILENT_MODE:
            print(f"=== Iteration {iteration} ===")

        for path, spec in current_ranges.items():
            if converged[path] and path in convergeable:
                if not SILENT_MODE:
                    print(f"Skipping {path}, already converged.")
                continue

            if isinstance(spec, list):
                cands = spec
            else:
                low, high = spec
                if isinstance(low, bool) and isinstance(high, bool):
                    cands = [False, True]
                else:
                    arr = np.linspace(low, high, num_candidates)
                    cands = arr.astype(int).tolist() if isinstance(low, int) and isinstance(high, int) else arr.tolist()

            eps_abs = relative_epsilon * (orig_ranges[path] or 1)
            val, score, conv = search_parameter(
                path, cands, best_params, cfg,
                train_loader, val_loader, num_classes, device,
                search_epochs, eps_abs, max_depth
            )

            if writer:
                depth = 0
                for cand in cands:
                    writer.writerow([*(best_params[p] if p != path else cand for p in param_grid)] + [depth, iteration, score])

            best_params[path] = val
            converged[path] = conv
            overall_best = max(overall_best, score)
            if not SILENT_MODE:
                print(f"-> {path}: best={val}, score={score:.4f}, conv={conv}")

            if not isinstance(spec, list) and not conv and isinstance(spec[0], (int, float)):
                low, high = spec
                w = high - low
                half = w / 2
                if val == low:
                    new_spec = (low, low + half)
                elif val == high:
                    new_spec = (high - half, high)
                else:
                    nl = max(val - half/2, low)
                    nh = min(val + half/2, high)
                    if nh - nl < half:
                        if nh < high:
                            nh = nl + half
                        else:
                            nl = nh - half
                    new_spec = (nl, nh)
                if all(isinstance(x, int) for x in spec):
                    new_spec = tuple(int(round(x)) for x in new_spec)
                current_ranges[path] = new_spec

        if not SILENT_MODE:
            print("--- End Iteration ---")
        if all(converged[p] for p in convergeable):
            break

    for path, val in best_params.items():
        _set_attr(cfg, path, val)

    if csv_file:
        csv_file.close()

    # Always print final config if silent or not
    print("Final config:", cfg)

    return best_params, overall_best
