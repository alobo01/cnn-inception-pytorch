#!/usr/bin/env python
"""
plots_from_logs.py — Generate figures that illustrate the optimisation process and
overall performance of every CNN model evaluated on MAMe, including accuracy and loss curves.

Example
-------
    python .\scripts\plots_from_logs.py logs\inception.log logs\standard.log logs\vgg19.log --outdir figs

The script produces, inside *figs/* by default:
    • acc_curve_<model>.png        – per-model train/val accuracy lines
    • loss_curve_<model>.png       – per-model train/val loss lines
    • val_acc_comparison.png       – single plot comparing validation accuracy
    • val_loss_comparison.png      – single plot comparing validation loss
    • best_hyperparams.csv         – comma-separated summary of the best H-P search results

Those artefacts are referenced by the LaTeX *Results* section.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# -----------------------------------------------------------------------------
# 1.  Regex helpers ------------------------------------------------------------
# -----------------------------------------------------------------------------
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+:\s+train_loss=([\d.]+)\s+acc=([\d.]+)\s+\|\s+val_loss=([\d.]+)\s+acc=([\d.]+)")
BEST_HP_RE = re.compile(r"Best hyperparameters:\s*(\{.*?\})")
BEST_VAL_RE = re.compile(r"Best validation score:\s*([\d.]+)")


def _parse_epoch_metrics(lines: List[str]) -> pd.DataFrame:
    """Return DataFrame with columns epoch, train_loss, train_acc, val_loss, val_acc."""
    rows: List[Dict[str, float]] = []
    for ln in lines:
        m = EPOCH_RE.search(ln)
        if m:
            ep, tl, ta, vl, va = m.groups()
            rows.append({
                "epoch": int(ep),
                "train_loss": float(tl),
                "train_acc": float(ta),
                "val_loss": float(vl),
                "val_acc": float(va),
            })
    return pd.DataFrame(rows)


def _parse_best(lines: List[str]) -> Dict[str, str | float]:
    """Return {hp: str, val_score: float}."""
    hp = next((BEST_HP_RE.search(ln).group(1) for ln in lines if BEST_HP_RE.search(ln)), "{}")
    val = next((float(BEST_VAL_RE.search(ln).group(1)) for ln in lines if BEST_VAL_RE.search(ln)), float("nan"))
    return {"hp": hp, "val_score": val}


# -----------------------------------------------------------------------------
# 2.  Main entry-point ---------------------------------------------------------
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate training curves and comparison plots from log files.")
    ap.add_argument("logs", nargs="+", help="*.log files (one per model)")
    ap.add_argument("--outdir", default="figs", help="Where to save the generated figures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    acc_frames: List[pd.DataFrame] = []
    best_rows: List[Dict[str, str | float]] = []

    for log_file in args.logs:
        model_name = Path(log_file).stem.replace(".log", "")
        with open(log_file) as fh:
            lines = fh.readlines()

        df = _parse_epoch_metrics(lines)
        df["model"] = model_name
        acc_frames.append(df)

        best = _parse_best(lines)
        best["model"] = model_name
        best_rows.append(best)

        # Per-model accuracy curves ------------------------------------------------
        fig, ax = plt.subplots()
        ax.plot(df["epoch"], df["train_acc"], label="train_acc")
        ax.plot(df["epoch"], df["val_acc"], label="val_acc")
        ax.set(xlabel="Epoch", ylabel="Accuracy", title=f"Accuracy – {model_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"acc_curve_{model_name}.png", dpi=300)
        plt.close(fig)

        # Per-model loss curves -----------------------------------------------------
        fig, ax = plt.subplots()
        ax.plot(df["epoch"], df["train_loss"], label="train_loss")
        ax.plot(df["epoch"], df["val_loss"], label="val_loss")
        ax.set(xlabel="Epoch", ylabel="Loss", title=f"Loss – {model_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"loss_curve_{model_name}.png", dpi=300)
        plt.close(fig)


    # Combine all epoch‐level metrics into one big DataFrame
    all_acc = pd.concat(acc_frames, ignore_index=True)

    # 1) Combined train & val loss across all models (existing) ----------------
    fig, ax = plt.subplots()
    for model, group in all_acc.groupby("model"):
        sns.lineplot(data=group, x="epoch", y="train_loss", ax=ax, label=f"{model} (train)")
        sns.lineplot(data=group, x="epoch", y="val_loss",   ax=ax, label=f"{model} (val)")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training & Validation Loss Comparison")
    fig.tight_layout()
    fig.savefig(outdir / "loss_comparison_all.png", dpi=300)
    plt.close(fig)

    # 2) Combined train & val accuracy across all models (existing) ------------
    fig, ax = plt.subplots()
    for model, group in all_acc.groupby("model"):
        sns.lineplot(data=group, x="epoch", y="train_acc", ax=ax, label=f"{model} (train)")
        sns.lineplot(data=group, x="epoch", y="val_acc",   ax=ax, label=f"{model} (val)")
    ax.set(xlabel="Epoch", ylabel="Accuracy", title="Training & Validation Accuracy Comparison")
    fig.tight_layout()
    fig.savefig(outdir / "acc_comparison_all.png", dpi=300)
    plt.close(fig)

    # ————————————————
    # 3) Train‐only loss comparison ---------------------------------------------
    fig, ax = plt.subplots()
    for model, group in all_acc.groupby("model"):
        sns.lineplot(data=group, x="epoch", y="train_loss", ax=ax, label=model)
    ax.set(xlabel="Epoch", ylabel="Loss", title="Train Loss Comparison")
    fig.tight_layout()
    fig.savefig(outdir / "train_loss_comparison_all.png", dpi=300)
    plt.close(fig)

    # 4) Val‐only loss comparison -----------------------------------------------
    fig, ax = plt.subplots()
    for model, group in all_acc.groupby("model"):
        sns.lineplot(data=group, x="epoch", y="val_loss", ax=ax, label=model)
    ax.set(xlabel="Epoch", ylabel="Loss", title="Validation Loss Comparison")
    fig.tight_layout()
    fig.savefig(outdir / "val_loss_comparison_all.png", dpi=300)
    plt.close(fig)

    # 5) Train‐only accuracy comparison -----------------------------------------
    fig, ax = plt.subplots()
    for model, group in all_acc.groupby("model"):
        sns.lineplot(data=group, x="epoch", y="train_acc", ax=ax, label=model)
    ax.set(xlabel="Epoch", ylabel="Accuracy", title="Train Accuracy Comparison")
    fig.tight_layout()
    fig.savefig(outdir / "train_acc_comparison_all.png", dpi=300)
    plt.close(fig)

    # 6) Val‐only accuracy comparison -------------------------------------------
    fig, ax = plt.subplots()
    for model, group in all_acc.groupby("model"):
        sns.lineplot(data=group, x="epoch", y="val_acc", ax=ax, label=model)
    ax.set(xlabel="Epoch", ylabel="Accuracy", title="Validation Accuracy Comparison")
    fig.tight_layout()
    fig.savefig(outdir / "val_acc_comparison_all.png", dpi=300)
    plt.close(fig)

    # CSV summary of best hyper-params -------------------------------------------
    pd.DataFrame(best_rows).to_csv(outdir / "best_hyperparams.csv", index=False)


if __name__ == "__main__":
    main()
