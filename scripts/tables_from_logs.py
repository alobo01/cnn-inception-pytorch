#!/usr/bin/env python
"""
tables_from_logs.py — Extract end‑of‑training metrics from *.log files and emit a
ready‑to‑\input{}‑able LaTeX table.

Example
-------
    python scripts/tables_from_logs.py logs/standard.log logs/inception.log logs/vgg19.log --outfile figs/results_table.tex
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Dict

import pandas as pd

TEST_RE = re.compile(r"Test loss:\s*([\d.]+)\s*\|\s*Test acc:\s*([\d.]+)")
BEST_RE = re.compile(r"Best validation score:\s*([\d.]+)")


def _parse_file(path: str | Path) -> Dict[str, float]:
    lines = Path(path).read_text().splitlines()
    test_loss = test_acc = best_val = float("nan")
    for ln in lines:
        if m := TEST_RE.search(ln):
            test_loss, test_acc = map(float, m.groups())
        elif m := BEST_RE.search(ln):
            best_val = float(m.group(1))
    return {
        "Test Loss": test_loss,
        "Test Acc": test_acc,
        "Best Val Acc": best_val,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert logs into a LaTeX metrics table.")
    ap.add_argument("logs", nargs="+", help="*.log files")
    ap.add_argument("--outfile", default="figs/results_table.tex")
    args = ap.parse_args()

    rows: List[Dict[str, float]] = []
    for p in args.logs:
        model = Path(p).stem
        metrics = _parse_file(p)
        metrics["Model"] = model
        rows.append(metrics)

    df = pd.DataFrame(rows)[["Model", "Best Val Acc", "Test Acc", "Test Loss"]]
    latex_str = df.to_latex(index=False, float_format="%.3f")
    Path(args.outfile).write_text(latex_str)
    print(f"Wrote LaTeX table → {args.outfile}")


if __name__ == "__main__":
    main()
