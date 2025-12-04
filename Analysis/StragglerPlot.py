"""
Plot test accuracy vs total_agg for selected straggler experiments (20/30/40/50_pct),
comparing FedBuff and TrustWeight.
Outputs one PDF per experiment under analysis/.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

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


def _find_csv(exp_dir: Path, prefix: str) -> Path | None:
    for candidate in sorted(exp_dir.glob(f"{prefix}*.csv")):
        if candidate.is_file():
            return candidate
    return None


def _read_curve(csv_path: Path, y_field: str = "test_acc") -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row.get("total_agg", "nan"))
                y = float(row.get(y_field, "nan"))
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


def plot_selected(
    exp_root: Path = Path("results/StragglerSweep"),
    experiments: Sequence[str] = ("20_pct", "30_pct", "40_pct", "50_pct"),
    out_dir: Path = Path("results/StragglerSweep"),
    outfile_suffix: str = "pdf",
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_w, fig_h = 8.0, 5.0

    saved: List[Path] = []
    for exp in experiments:
        exp_dir = exp_root / exp
        if not exp_dir.exists():
            print(f"[analysis] Missing experiment dir: {exp_dir}")
            continue

        fb_csv = _find_csv(exp_dir, "FedBuff")
        tw_csv = _find_csv(exp_dir, "TrustWeight")

        curves: List[Tuple[str, List[float], List[float]]] = []
        for label, path in (("FedBuff", fb_csv), ("TrustWeight", tw_csv)):
            if path is None:
                print(f"[analysis] Missing {label} CSV in {exp_dir}")
                continue
            xs, ys = _read_curve(path)
            if not xs:
                print(f"[analysis] No data in {path}")
                continue
            curves.append((label, xs, ys))

        if not curves:
            print(f"[analysis] No plottable data for {exp_dir}")
            continue

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        color_map = {"FedBuff": "royalblue", "TrustWeight": "crimson"}
        for label, xs, ys in curves:
            ax.plot(xs, ys, label=label, linewidth=1.8, color=color_map.get(label))

        ax.set_xlabel("Total aggregations")
        ax.set_ylabel("Test accuracy")
        ax.set_title(exp.replace("_pct", "% stragglers"))
        ax.grid(True, color="gray", linestyle="--", linewidth=0.2, alpha=0.5)
        ax.legend(frameon=False, loc="upper left")

        out_path = out_dir / f"stag_{exp}.{outfile_suffix}"
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, format=outfile_suffix, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        print(f"[analysis] Saved plot to {out_path}")

    return saved


if __name__ == "__main__":
    plot_selected()
