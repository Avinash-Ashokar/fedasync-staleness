"""
Plot test accuracy vs total_agg for the alpha experiments, comparing
FedBuff and TrustWeight curves stored under logs/alpha/.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

# Visual style consistent with other analysis plots
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.prop_cycle": cycler(color=["royalblue", "crimson"]),
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)


def _read_curve(csv_path: Path) -> Tuple[List[float], List[float]]:
    """Return total_agg and test_acc lists from a CSV file."""
    xs: List[float] = []
    ys: List[float] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row.get("total_agg", "nan"))
                y = float(row.get("test_acc", "nan"))
            except Exception:
                continue
            if math.isfinite(x) and math.isfinite(y):
                xs.append(x)
                ys.append(y)
    if not xs:
        return xs, ys
    pairs = sorted(zip(xs, ys), key=lambda p: p[0])
    xs_sorted, ys_sorted = zip(*pairs)
    return list(xs_sorted), list(ys_sorted)


def plot_alpha_compare(
    exp_root: Path = Path("results/Accuracy"),
    out_dir: Path = Path("results/Accuracy"),
    outfile: str = "accuracy.pdf",
) -> Path:
    """Plot FedBuff vs TrustWeight test accuracy for the alpha experiment set."""
    fb_csv = next((p for p in exp_root.glob("FedBuff*.csv") if p.is_file()), None)
    tw_csv = next((p for p in exp_root.glob("TrustWeight*.csv") if p.is_file()), None)

    if fb_csv is None and tw_csv is None:
        print(f"[analysis] No FedBuff/TrustWeight CSVs found under {exp_root}")
        return out_dir / outfile

    curves: List[Tuple[str, List[float], List[float]]] = []
    if fb_csv:
        xs, ys = _read_curve(fb_csv)
        if xs:
            curves.append(("FedBuff", xs, ys))
        else:
            print(f"[analysis] No data in {fb_csv}")
    if tw_csv:
        xs, ys = _read_curve(tw_csv)
        if xs:
            curves.append(("TrustWeight", xs, ys))
        else:
            print(f"[analysis] No data in {tw_csv}")

    if not curves:
        print(f"[analysis] No plottable data under {exp_root}")
        return out_dir / outfile

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_w, fig_h = 8.0, 5.0  # inches
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    color_map = {"FedBuff": "royalblue", "TrustWeight": "crimson"}
    for label, xs, ys in curves:
        ax.plot(xs, ys, label=label, linewidth=1.8, color=color_map.get(label))

    ax.set_xlabel("Total aggregations")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Alpha Experiment")
    ax.grid(True, color="gray", linestyle="--", linewidth=0.2, alpha=0.5)
    ax.legend(frameon=False, loc="lower right")

    out_path = out_dir / outfile
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[analysis] Saved plot to {out_path}")
    return out_path


if __name__ == "__main__":
    plot_alpha_compare()
