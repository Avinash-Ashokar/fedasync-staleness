# Straggler sweep runner for FedAsync
import os
import time
import random
import logging
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List
from concurrent.futures import ThreadPoolExecutor

import yaml

from FedAsync.client import LocalAsyncClient
from FedAsync.server import AsyncFedServer
from utils.model import build_resnet18
from utils.partitioning import DataDistributor
from utils.helper import set_seed, get_device


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _override_io(cfg: Dict[str, Any], exp_dir: Path) -> Dict[str, Any]:
    cfg = deepcopy(cfg)
    exp_dir = exp_dir.resolve()
    cfg["io"]["logs_dir"] = str(exp_dir)
    cfg["io"]["checkpoints_dir"] = str(exp_dir / "checkpoints")
    cfg["io"]["results_dir"] = str(exp_dir / "results")
    cfg["io"]["global_log_csv"] = str(exp_dir / "FedAsync.csv")
    cfg["io"]["client_participation_csv"] = str(exp_dir / "FedAsyncClientParticipation.csv")
    cfg["io"]["final_model_path"] = str(exp_dir / "results" / "FedAsyncModel.pt")
    return cfg


def run_once(cfg: Dict[str, Any]) -> None:
    # Silence noisy logs
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["LIGHTNING_DISABLE_RICH"] = "1"
    for name in [
        "pytorch_lightning", "lightning", "lightning.pytorch",
        "lightning_fabric", "lightning_utilities", "torch", "torchvision",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)
        logging.getLogger(name).propagate = False
    logging.getLogger().setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")

    # Reproducibility
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    random.seed(seed)

    # Partition dataset
    dd = DataDistributor(dataset_name=cfg["data"]["dataset"], data_dir=cfg["data"]["data_dir"])
    dd.distribute_data(
        num_clients=int(cfg["clients"]["total"]),
        alpha=float(cfg.get("partition_alpha", 0.5)),
        seed=seed,
    )

    # Build server
    global_model = build_resnet18(num_classes=cfg["data"]["num_classes"], pretrained=False)
    server = AsyncFedServer(
        global_model=global_model,
        alpha=float(cfg["async"]["alpha"]),
        target_accuracy=float(cfg["eval"]["target_accuracy"]),
        max_rounds=int(cfg["train"]["max_rounds"]) if "max_rounds" in cfg["train"] else None,
        eval_every_aggs=int(cfg["eval"].get("eval_every_aggs", 5)),
        data_dir=cfg["data"]["data_dir"],
        logs_dir=cfg["io"]["logs_dir"],
        global_log_csv=cfg["io"].get("global_log_csv"),
        client_participation_csv=cfg["io"].get("client_participation_csv"),
        final_model_path=cfg["io"].get("final_model_path"),
        num_classes=int(cfg["data"]["num_classes"]),
        device=get_device(),
    )

    # Straggler sampling
    n = int(cfg["clients"]["total"])
    pct = max(0, min(100, int(cfg["clients"].get("struggle_percent", 0))))
    k_slow = (n * pct) // 100
    slow_ids = set(random.sample(range(n), k_slow)) if k_slow > 0 else set()

    a_s, b_s = cfg["clients"].get("delay_slow_range", [0.8, 2.0])
    a_f, b_f = cfg["clients"].get("delay_fast_range", [0.0, 0.2])
    fix_delays = bool(cfg["clients"].get("fix_delays_per_client", True))
    jitter = float(cfg["clients"].get("jitter_per_round", 0.0))

    per_client_base_delay: Dict[int, float] = {}
    if fix_delays:
        for cid in range(n):
            if cid in slow_ids:
                per_client_base_delay[cid] = random.uniform(float(a_s), float(b_s))
            else:
                per_client_base_delay[cid] = random.uniform(float(a_f), float(b_f))

    # Clients
    clients: List[LocalAsyncClient] = []
    for cid in range(n):
        subset = dd.get_client_data(cid)
        base_delay = per_client_base_delay.get(cid, 0.0)
        is_slow = cid in slow_ids
        clients.append(LocalAsyncClient(
            cid=cid,
            cfg=cfg,
            subset=subset,
            base_delay=base_delay,
            slow=is_slow,
            delay_ranges=((float(a_s), float(b_s)), (float(a_f), float(b_f))),
            jitter=jitter,
            fix_delay=fix_delays,
        ))

    def client_loop(client: LocalAsyncClient):
        try:
            while not server.should_stop():
                cont = client.fit_once(server)
                if not cont:
                    break
                time.sleep(0.05)
        except Exception:
            server.mark_stop()
            raise

    with ThreadPoolExecutor(max_workers=int(cfg["clients"]["concurrent"])) as executor:
        futures = [executor.submit(client_loop, cl) for cl in clients]
        try:
            while not server.should_stop():
                if all(f.done() for f in futures):
                    server.mark_stop()
                    break
                time.sleep(0.2)
        finally:
            server.mark_stop()
            for f in futures:
                f.result()


def straggler_sweep(
    base_cfg: Dict[str, Any],
    percents: Iterable[int],
    out_root: Path,
) -> None:
    for pct in percents:
        exp_dir = out_root / f"straggle_{pct}pct"
        cfg = deepcopy(base_cfg)
        cfg["clients"]["struggle_percent"] = int(pct)
        cfg = _override_io(cfg, exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"[straggler_sweep] percent={pct}% -> logs at {exp_dir}")
        run_once(cfg)


if __name__ == "__main__":
    cfg_path = os.environ.get("FEDASYNC_CONFIG", os.path.join(os.path.dirname(__file__), "config.yaml"))
    base = load_cfg(cfg_path)
    out_root = Path(base["io"]["logs_dir"]) / "FedAsyncStragglerExp"
    out_root.mkdir(parents=True, exist_ok=True)
    straggler_sweep(base, percents=[0, 10, 20, 30, 40, 50], out_root=out_root)
