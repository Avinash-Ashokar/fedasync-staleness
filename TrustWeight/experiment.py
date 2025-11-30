# Experiment runner for TrustWeight sweeps (partition alpha and straggler percent)
import logging
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import math
import csv
from typing import Iterable, List
import threading

from .config import GlobalConfig, load_config
from .run import _set_seed
from .client import AsyncClient
from .server import AsyncServer
from utils.partitioning import DataDistributor


def _safe_name(val) -> str:
    """Convert numeric value to a filesystem-friendly token."""
    s = str(val)
    return s.replace(".", "p")


def _override_io(cfg: GlobalConfig, exp_dir: Path) -> GlobalConfig:
    """Return a copy of cfg with I/O paths pointing inside exp_dir."""
    exp_dir = exp_dir.resolve()
    io = cfg.io
    new_io = replace(
        io,
        logs_dir=str(exp_dir),
        checkpoints_dir=str(exp_dir / "checkpoints"),
        results_dir=str(exp_dir / "results"),
        global_log_csv=str(exp_dir / "TrustWeight.csv"),
        client_participation_csv=str(exp_dir / "TrustWeightClientParticipation.csv"),
        final_model_path=str(exp_dir / "TrustWeightModel.pt"),
    )
    return replace(cfg, io=new_io)


def run_with_cfg(cfg: GlobalConfig) -> None:
    """Run one TrustWeight training session with an already-mutated cfg."""
    # --------------------- seed ---------------------
    logging.basicConfig(level=logging.INFO)
    _set_seed(cfg.seed)

    # --------------------- dataset & partition --------------------
    distributor = DataDistributor(
        dataset_name=cfg.data.dataset,
        data_dir=cfg.data.data_dir,
    )
    distributor.distribute_data(
        num_clients=cfg.clients.total,
        alpha=cfg.partition_alpha,
        seed=cfg.seed,
    )
    partitions = [distributor.partitions[cid] for cid in range(cfg.clients.total)]

    # --------------------- create server -------------------------
    server = AsyncServer(cfg=cfg)

    # --------------------- create clients ------------------------
    clients: List[AsyncClient] = []
    for cid in range(cfg.clients.total):
        indices = partitions[cid] if cid < len(partitions) else []
        clients.append(AsyncClient(cid=cid, indices=indices, cfg=cfg))

    # ------------------- concurrency control ---------------------
    sem = threading.Semaphore(cfg.clients.concurrent)

    def client_loop(cl: AsyncClient) -> None:
        while not server.should_stop():
            with sem:
                cont = cl.run_once(server)
            if not cont or server.should_stop():
                break

    # --------------------- start client threads ------------------
    threads: List[threading.Thread] = []
    for cl in clients:
        t = threading.Thread(target=client_loop, args=(cl,), daemon=True)
        t.start()
        threads.append(t)

    # --------------------- wait for completion -------------------
    server.wait()
    for t in threads:
        t.join(timeout=1.0)


def alpha_sweep(
    base_cfg: GlobalConfig,
    alphas: Iterable[float],
    out_root: Path,
) -> None:
    """Run multiple trainings varying the Dirichlet alpha."""
    for alpha in alphas:
        exp_dir = out_root / f"alpha_{_safe_name(alpha)}"
        cfg = deepcopy(base_cfg)
        cfg.partition_alpha = float(alpha)
        cfg = _override_io(cfg, exp_dir)
        print(f"[exp] alpha={alpha} -> logs at {exp_dir}")
        run_with_cfg(cfg)


def straggler_sweep(
    base_cfg: GlobalConfig,
    percents: Iterable[int],
    out_root: Path,
) -> None:
    """Run multiple trainings varying the percent of slow clients."""
    for pct in percents:
        exp_dir = out_root / f"straggle_{pct}pct"
        cfg = deepcopy(base_cfg)
        cfg.clients.struggle_percent = int(pct)
        cfg = _override_io(cfg, exp_dir)
        print(f"[exp] struggle_percent={pct}% -> logs at {exp_dir}")
        run_with_cfg(cfg)


def sanity_check(exp_root: Path) -> None:
    """Lightweight sanity check over all experiment folders.

    Confirms required CSVs exist and contain finite test_acc values.
    Prints a short summary of best/last test_acc for quick inspection.
    """
    exp_root = exp_root.resolve()
    if not exp_root.exists():
        print(f"[sanity] No experiment folder found at {exp_root}")
        return

    exp_dirs = [p for p in exp_root.iterdir() if p.is_dir()]
    if not exp_dirs:
        print(f"[sanity] No sub-experiments found in {exp_root}")
        return

    for exp in sorted(exp_dirs):
        global_csv = exp / "TrustWeight.csv"
        client_csv = exp / "TrustWeightClientParticipation.csv"

        missing = [p.name for p in [global_csv, client_csv] if not p.exists()]
        if missing:
            print(f"[sanity] {exp.name}: missing {missing}")
            continue

        try:
            rows = []
            with global_csv.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
        except Exception as e:
            print(f"[sanity] {exp.name}: failed to read CSV ({e})")
            continue

        if not rows:
            print(f"[sanity] {exp.name}: global CSV is empty")
            continue

        def _safe_float(x):
            try:
                v = float(x)
                return v if math.isfinite(v) else None
            except Exception:
                return None

        test_accs = [_safe_float(r.get("test_acc")) for r in rows]
        test_accs = [v for v in test_accs if v is not None]

        if not test_accs:
            print(f"[sanity] {exp.name}: no finite test_acc values")
            continue

        best = max(test_accs)
        last = test_accs[-1]
        print(f"[sanity] {exp.name}: best_test_acc={best:.4f}, last_test_acc={last:.4f}, n_rows={len(rows)}")


if __name__ == "__main__":
    base = load_config()
    out_root = Path(base.io.logs_dir) / "TrustWeightDataExp"
    out_root.mkdir(parents=True, exist_ok=True)

    sta_root = Path(base.io.logs_dir) / "TrustWeightStragglerExp"
    sta_root.mkdir(parents=True, exist_ok=True)

    # Experiment 1: Dirichlet alpha sweep (non-IID -> IID)
    alpha_sweep(base, alphas=[1, 10, 100, 1000], out_root=out_root)

    # Experiment 2: straggler percentage sweep
    straggler_sweep(base, percents=[10, 20, 30, 40, 50], out_root=sta_root)

