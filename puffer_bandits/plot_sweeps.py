from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore


def load_rows(path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def aggregate(rows: list[dict[str, str]], algo: str, param: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Group by parameter value
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("algo") != algo:
            continue
        key = row.get(param)
        if key is None:
            continue
        try:
            val = float(row.get("final_mean_reward", "nan"))
        except Exception:
            continue
        if math.isnan(val):
            continue
        groups[key].append(val)
    # Sort by numeric param if possible
    def keyfn(k: str):
        try:
            return float(k)
        except Exception:
            return k
    keys = sorted(groups.keys(), key=keyfn)
    means = np.array([np.mean(groups[k]) for k in keys], dtype=float)
    ses = np.array([np.std(groups[k], ddof=1) / max(1, math.sqrt(len(groups[k]))) for k in keys], dtype=float)
    return np.array(keys), means, ses


def plot_line(keys: np.ndarray, means: np.ndarray, ses: np.ndarray, title: str, xlabel: str, outpath: str) -> None:
    if plt is None:
        print("matplotlib not available; skipping plot")
        return
    x = np.arange(len(keys))
    plt.figure(figsize=(6, 4))
    plt.plot(x, means, marker="o", label="final mean reward")
    lo = means - 1.96 * ses
    hi = means + 1.96 * ses
    plt.fill_between(x, lo, hi, alpha=0.2, label="95% CI")
    plt.xticks(x, [str(k) for k in keys])
    plt.xlabel(xlabel)
    plt.ylabel("Final mean reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser("Plot sweeps from CSV summaries")
    ap.add_argument("--file", type=str, required=True, help="Path to CSV (summary.csv or advanced_summary.csv)")
    ap.add_argument("--algo", type=str, required=True)
    ap.add_argument("--param", type=str, required=True, help="Column to group by (e.g., kl_alpha, window, alpha, features)")
    ap.add_argument("--out", type=str, default="plots_gpu/sweep_plot.png")
    args = ap.parse_args()

    rows = load_rows(args.file)
    keys, means, ses = aggregate(rows, args.algo, args.param)
    title = f"{args.algo} sweep over {args.param}"
    xlabel = args.param
    plot_line(keys, means, ses, title, xlabel, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

