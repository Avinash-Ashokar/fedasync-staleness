"""Grid search for TrustWeightedAsyncStrategy hyperparameters (beta1, beta2, theta)."""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch

from TrustWeight.client import AsyncClient
from TrustWeight.config import GlobalConfig, load_config
from TrustWeight.run import _set_seed
from TrustWeight.server import AsyncServer
from TrustWeight.strategy import TrustWeightedConfig
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


def _format_theta(theta: Sequence[float]) -> str:
    if len(theta) != 3:
        raise ValueError("theta must have 3 values: (delta_loss, norm_u, cos)")
    return "_".join(_safe_name(t) for t in theta)


def _parse_theta(arg: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in arg.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("theta must be 'a,b,c'")
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def _apply_strategy_config(server: AsyncServer, cfg: TrustWeightedConfig) -> None:
    """Inject strategy hyperparameters into an existing server instance."""
    server.strategy.cfg = cfg
    server.strategy.theta = torch.tensor(cfg.theta, dtype=torch.float32)
    # Reset momentum to avoid bleed-over between runs when reusing code paths
    server.strategy.m = torch.zeros_like(server.strategy.m)


def run_with_strategy_cfg(cfg: GlobalConfig, strategy_cfg: TrustWeightedConfig) -> None:
    """Run one TrustWeight training session with custom strategy settings."""
    logging.basicConfig(level=logging.INFO)
    _set_seed(cfg.seed)

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

    server = AsyncServer(cfg=cfg)
    _apply_strategy_config(server, strategy_cfg)

    clients: List[AsyncClient] = []
    for cid in range(cfg.clients.total):
        indices = partitions[cid] if cid < len(partitions) else []
        clients.append(AsyncClient(cid=cid, indices=indices, cfg=cfg))

    def client_loop(cl: AsyncClient) -> None:
        while not server.should_stop():
            cont = cl.run_once(server)
            if not cont or server.should_stop():
                break

    with ThreadPoolExecutor(max_workers=cfg.clients.concurrent) as executor:
        futures = [executor.submit(client_loop, cl) for cl in clients]

        server.wait()
        for f in futures:
            try:
                f.result(timeout=1.0)
            except Exception:
                pass


def grid_search_strategy(
    base_cfg: GlobalConfig,
    beta1_values: Iterable[float],
    beta2_values: Iterable[float],
    theta_values: Iterable[Sequence[float]],
    out_root: Path,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for beta1, beta2, theta in product(beta1_values, beta2_values, theta_values):
        theta_tuple = tuple(float(x) for x in theta)
        tw_cfg = TrustWeightedConfig(beta1=float(beta1), beta2=float(beta2), theta=theta_tuple)
        exp_name = (
            f"b1_{_safe_name(beta1)}__"
            f"b2_{_safe_name(beta2)}__"
            f"theta_{_format_theta(theta_tuple)}"
        )
        exp_dir = out_root / exp_name
        cfg = _override_io(deepcopy(base_cfg), exp_dir)
        print(f"[grid] beta1={beta1}, beta2={beta2}, theta={theta_tuple} -> {exp_dir}")
        run_with_strategy_cfg(cfg, tw_cfg)


def _default_theta_grid() -> List[Tuple[float, float, float]]:
    return [
        (0.0, 0.0, 0.0),
        (0.1, -0.05, 0.0),
        (0.2, -0.1, 0.05),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search beta1/beta2/theta for TrustWeight")
    parser.add_argument("--config", type=str, default="TrustWeight/config.yaml", help="Path to base config.yaml")
    parser.add_argument("--beta1", type=float, nargs="+", default=[0.0, 0.1, 0.2], help="List of beta1 values")
    parser.add_argument("--beta2", type=float, nargs="+", default=[0.0, 0.05, 0.1], help="List of beta2 values")
    parser.add_argument(
        "--theta",
        type=_parse_theta,
        nargs="+",
        default=_default_theta_grid(),
        help="List of theta triples 'a,b,c'",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Override output root (defaults to <logs_dir>/TrustWeightStrategyGrid)",
    )

    args = parser.parse_args()

    base_cfg = load_config(args.config)
    out_root = Path(args.out_root) if args.out_root else Path(base_cfg.io.logs_dir) / "TrustWeightStrategyGrid"
    grid_search_strategy(
        base_cfg=base_cfg,
        beta1_values=args.beta1,
        beta2_values=args.beta2,
        theta_values=args.theta,
        out_root=out_root,
    )


if __name__ == "__main__":
    main()
