import argparse
from pathlib import Path

import torch
import numpy as np

from utils.config import Config
from utils.dataset import create_data_loaders
from utils.training_helpers import _get_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model without sklearn")
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to YAML config for model instantiation"
    )
    parser.add_argument(
        "--weights", type=Path, required=True,
        help="Path to .pth file with trained model weights"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output folder for results (default: results/<model_type>)"
    )
    args = parser.parse_args()

    # Load config and prepare output directory
    cfg = Config.load(args.config)
    model_type = cfg.model.type
    out_dir = args.output or Path("results") / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data loader
    (_, _, test_loader), classes = create_data_loaders(cfg)
    if classes is None:
        num_classes = cfg.model.num_classes if hasattr(cfg.model, 'num_classes') else 0
        classes = [str(i) for i in range(num_classes)]
    num_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model and load weights
    model = _get_model(cfg, num_classes=num_classes).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Gather predictions and targets
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Compute confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    # Compute per-class metrics
    per_class = []
    precisions, recalls, f1s, supports = [], [], [], []
    for idx, cls in enumerate(classes):
        TP = cm[idx, idx]
        FP = cm[:, idx].sum() - TP
        FN = cm[idx, :].sum() - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = cm[idx, :].sum()
        precisions.append(precision); recalls.append(recall)
        f1s.append(f1); supports.append(support)
        per_class.append((cls, precision, recall, f1, support))

    # Aggregate metrics
    accuracy = cm.diagonal().sum() / cm.sum()
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    # Write plain text metrics and CSV
    lines = [
        f"Accuracy: {accuracy:.4f}",
        f"Macro Precision: {macro_precision:.4f}",
        f"Macro Recall: {macro_recall:.4f}",
        f"Macro F1-score: {macro_f1:.4f}\n",
        "Class | Precision | Recall | F1-score | Support",
        "----- | --------- | ------ | -------- | -------"
    ]
    for cls, prec, rec, f1, sup in per_class:
        lines.append(f"{cls} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {sup}")
    (out_dir / "metrics.txt").write_text("\n".join(lines))
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt='%d', delimiter=',')

    # Write LaTeX tables
    # Classification metrics table
    metrics_tex = []
    metrics_tex.append(r"\begin{table}[ht]")
    metrics_tex.append(r"\centering")
    metrics_tex.append(r"\begin{tabular}{lrrrr}")
    metrics_tex.append(r"\hline")
    metrics_tex.append(r"Class & Precision & Recall & F1-score & Support \\")
    metrics_tex.append(r"\hline")
    for cls, prec, rec, f1, sup in per_class:
        metrics_tex.append(f"{cls} & {prec:.4f} & {rec:.4f} & {f1:.4f} & {sup} \\")
    metrics_tex.append(r"\hline")
    metrics_tex.append(r"\end{tabular}")
    metrics_tex.append(r"\caption{Classification report metrics}")
    metrics_tex.append(r"\end{table}")
    (out_dir / "metrics.tex").write_text("\n".join(metrics_tex))

    # Confusion matrix table
    cm_tex = []
    cm_tex.append(r"\begin{table}[ht]")
    cm_tex.append(r"\centering")
    col_fmt = "l" + "r" * num_classes
    cm_tex.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    cm_tex.append(r"\hline")
    header = " & " + " & ".join(classes) + r" \\"
    cm_tex.append(header)
    cm_tex.append(r"\hline")
    for i, cls in enumerate(classes):
        row = [cls] + [str(cm[i, j]) for j in range(num_classes)]
        cm_tex.append(" & ".join(row) + r" \\")
    cm_tex.append(r"\hline")
    cm_tex.append(r"\end{tabular}")
    cm_tex.append(r"\caption{Confusion matrix}")
    cm_tex.append(r"\end{table}")
    (out_dir / "confusion_matrix.tex").write_text("\n".join(cm_tex))

    print(f"Results saved to {out_dir} (metrics.txt, confusion_matrix.csv, metrics.tex, confusion_matrix.tex)")


if __name__ == "__main__":
    main()
