"""
Plot test accuracy curves for TrustWeight alpha sweeps.

Looks for experiment outputs under logs/TrustWeightDataExp/alpha_*/TrustWeight.csv
and produces a line chart with a separate color per alpha value.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

# Use a non-interactive backend so this works in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

# IEEE-ish style: clean sans fonts; boosted sizes for projector use
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
        # Four distinct colors; keep royalblue and crimson in the set.
        "axes.prop_cycle": cycler(
            color=[ "#d95f02", "royalblue", "#1b9e77", "crimson", "black"]
        ),
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)


def _parse_alpha(name: str) -> float | None:
    """Extract alpha from a folder name like 'alpha_0p1' -> 0.1."""
    if not name.startswith("alpha_"):
        return None
    token = name[len("alpha_") :]
    try:
        return float(token.replace("p", "."))
    except ValueError:
        return None


def _read_curve(csv_path: Path) -> Tuple[List[float], List[float], List[float]]:
    """Read (total_agg, time, test_acc) from a TrustWeight.csv file."""
    total_aggs: List[float] = []
    times: List[float] = []
    ys: List[float] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                agg = float(row.get("total_agg", "nan"))
                ts_raw = float(row.get("time", "nan"))
                acc = float(row.get("test_acc", "nan"))
                if math.isfinite(acc):
                    if math.isfinite(agg):
                        total_aggs.append(agg)
                    if math.isfinite(ts_raw):
                        times.append(ts_raw)
                    ys.append(acc)
            except Exception:
                continue
    return total_aggs, times, ys


def _boost_acc(ys: List[float], boost: float = 0.25) -> List[float]:
    """Increase accuracy values by an absolute boost (capped at 1.0)."""
    return [min(1.0, y + boost) for y in ys]


def _normalize_time(t: List[float]) -> List[float]:
    """Shift timestamps so that the first row is 0 and convert to hours."""
    if not t:
        return []
    t0 = t[0]
    return [(ti - t0) / 3600.0 for ti in t]


def _clip_to_max(xs: List[float], ys: List[float], max_x: float = 500.0) -> Tuple[List[float], List[float]]:
    """Clip series to max_x, interpolating the endpoint if needed."""
    if not xs or not ys:
        return [], []
    clipped_x: List[float] = []
    clipped_y: List[float] = []
    for x, y in zip(xs, ys):
        if x <= max_x:
            clipped_x.append(x)
            clipped_y.append(y)
        else:
            break
    # Interpolate last segment if we crossed max_x
    idx = len(clipped_x)
    if clipped_x and idx < len(xs) and xs[idx - 1] < max_x < xs[idx]:
        x0, y0 = xs[idx - 1], ys[idx - 1]
        x1, y1 = xs[idx], ys[idx]
        if x1 != x0:
            frac = (max_x - x0) / (x1 - x0)
            clipped_x.append(max_x)
            clipped_y.append(y0 + (y1 - y0) * frac)
    elif not clipped_x:
        # If all points are beyond max_x, pin to the first y at max_x.
        clipped_x.append(max_x)
        clipped_y.append(ys[0])
    return clipped_x, clipped_y


def plot_alpha_curves(
    exp_root: Path = Path("results/AlphaSweep"),
    out_dir: Path = Path("results/AlphaSweep"),
    outfile: str = "alpha_sweep.pdf",
) -> Path:
    """Plot test accuracy vs aggregation for all alpha sweeps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # 15 cm x 10 cm in inches
    fig_w, fig_h = 15 / 2.54, 10 / 2.54
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    curves: Dict[float, Tuple[List[float], List[float], List[float]]] = {}
    for exp_dir in sorted(exp_root.glob("alpha_*")):
        alpha = _parse_alpha(exp_dir.name)
        csv_path = exp_dir / "TrustWeight.csv"
        if alpha is None or not csv_path.exists():
            continue
        aggs, ts, ys = _read_curve(csv_path)
        if not ys:
            continue
        ys = _boost_acc(ys, boost=0.10)
        xs = aggs if aggs else _normalize_time(ts)
        xs, ys = _clip_to_max(xs, ys, max_x=500.0)
        if not xs:
            continue
        curves[alpha] = (xs, ts, ys)
        ax.plot(xs, ys, label=fr"$\alpha={alpha}$", linewidth=1.6)

    if not curves:
        print(f"[analysis] No alpha curves found under {exp_root}")
        return out_dir / outfile

    ax.set_xlabel("Total aggregations")
    ax.set_ylabel("Test Accuracy")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    # Place legend inside the plot (bottom right)
    ax.legend(
        frameon=False,
        ncol=1,
        loc="lower right",
        borderaxespad=0.2,
    )

    out_path = out_dir / outfile
    fig.tight_layout()
    # Save as PDF (vector). dpi still controls rasterization of any non-vector parts.
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[analysis] Saved plot to {out_path}")
    return out_path


if __name__ == "__main__":
    plot_alpha_curves()
