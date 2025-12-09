import logging
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import math
import csv
from typing import Iterable, List
import threading

from .config import GlobalConfig, load_config
from utils.helper import set_seed
from .client import AsyncClient
from .server import AsyncServer
from utils.partitioning import DataDistributor


def _safe_name(val) -> str:
    s = str(val)
    return s.replace(".", "p")


def _override_io(cfg: GlobalConfig, exp_dir: Path) -> GlobalConfig:
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
    logging.basicConfig(level=logging.INFO)
    set_seed(cfg.seed)

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

    from utils.model import build_resnet18
    global_model = build_resnet18(num_classes=cfg.data.num_classes, pretrained=False)
    server = AsyncServer(
        global_model=global_model,
        total_train_samples=len(distributor.train_dataset),
        buffer_size=5,
        buffer_timeout_s=5.0,
        use_sample_weighing=True,
        target_accuracy=cfg.eval.target_accuracy,
        max_rounds=cfg.train.max_rounds,
        eval_interval_s=cfg.eval.interval_seconds,
        data_dir=cfg.data.data_dir,
        checkpoints_dir=cfg.io.checkpoints_dir,
        logs_dir=cfg.io.logs_dir,
        global_log_csv=cfg.io.global_log_csv,
        client_participation_csv=cfg.io.client_participation_csv,
        final_model_path=cfg.io.final_model_path,
        resume=False,
        device=None,
        eta=1.0,
        theta=(1.0, -0.1, 0.2),
        freshness_alpha=0.1,
        beta1=0.0,
        beta2=0.0,
        momentum_gamma=0.9,
        update_clip_norm=cfg.train.update_clip_norm,
    )

    clients: List[AsyncClient] = []
    for cid in range(cfg.clients.total):
        indices = partitions[cid] if cid < len(partitions) else []
        clients.append(AsyncClient(cid=cid, indices=indices, cfg={
            "data": {"data_dir": cfg.data.data_dir, "num_classes": cfg.data.num_classes},
            "clients": {
                "total": cfg.clients.total,
                "batch_size": cfg.clients.batch_size,
                "local_epochs": cfg.clients.local_epochs,
                "lr": cfg.clients.lr,
                "weight_decay": cfg.clients.weight_decay,
                "grad_clip": cfg.clients.grad_clip,
                "struggle_percent": cfg.clients.struggle_percent,
                "delay_slow_range": list(cfg.clients.delay_slow_range),
                "delay_fast_range": list(cfg.clients.delay_fast_range),
                "jitter_per_round": cfg.clients.jitter_per_round,
            },
            "server_runtime": {"client_delay": cfg.server_runtime.client_delay},
        }))

    sem = threading.Semaphore(cfg.clients.concurrent)

    def client_loop(cl: AsyncClient) -> None:
        while not server.should_stop():
            with sem:
                cont = cl.run_once(server)
            if not cont or server.should_stop():
                break

    threads: List[threading.Thread] = []
    for cl in clients:
        t = threading.Thread(target=client_loop, args=(cl,), daemon=True)
        t.start()
        threads.append(t)

    server.wait()
    for t in threads:
        t.join(timeout=1.0)


def alpha_sweep(
    base_cfg: GlobalConfig,
    alphas: Iterable[float],
    out_root: Path,
) -> None:
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
    for pct in percents:
        exp_dir = out_root / f"straggle_{pct}pct"
        cfg = deepcopy(base_cfg)
        cfg.clients.struggle_percent = int(pct)
        cfg = _override_io(cfg, exp_dir)
        print(f"[exp] struggle_percent={pct}% -> logs at {exp_dir}")
        run_with_cfg(cfg)


def sanity_check(exp_root: Path) -> None:
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
