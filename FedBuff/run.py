import os
import logging, warnings
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

import threading
import time
from typing import Dict, Any, List
import random

import yaml

from FedBuff.client import LocalBuffClient
from FedBuff.server import BufferedFedServer
from utils.model import build_resnet18
from utils.partitioning import DataDistributor
from utils.helper import set_seed, get_device


CFG_PATH = os.environ.get("FEDBUFF_CONFIG", os.path.join(os.path.dirname(__file__), "config.yml"))


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_cfg(CFG_PATH)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    random.seed(seed)

    dd = DataDistributor(dataset_name=cfg["data"]["dataset"], data_dir=cfg["data"]["data_dir"])
    dd.distribute_data(
        num_clients=int(cfg["clients"]["total"]),
        alpha=float(cfg.get("partition_alpha", 0.5)),
        seed=seed,
    )

    global_model = build_resnet18(num_classes=cfg["data"]["num_classes"], pretrained=False)
    server = BufferedFedServer(
        global_model=global_model,
        total_train_samples=len(dd.train_dataset),
        buffer_size=int(cfg["buff"]["buffer_size"]),
        buffer_timeout_s=float(cfg["buff"]["buffer_timeout_s"]),
        use_sample_weighing=bool(cfg["buff"]["use_sample_weighing"]),
        target_accuracy=float(cfg["eval"]["target_accuracy"]),
        max_rounds=int(cfg["train"]["max_rounds"]) if "max_rounds" in cfg["train"] else None,
        eval_interval_s=int(cfg["eval"]["interval_seconds"]),
        data_dir=cfg["data"]["data_dir"],
        checkpoints_dir=cfg["io"]["checkpoints_dir"],
        logs_dir=cfg["io"]["logs_dir"],
        global_log_csv=cfg["io"].get("global_log_csv"),
        client_participation_csv=cfg["io"].get("client_participation_csv"),
        final_model_path=cfg["io"].get("final_model_path"),
        resume=True,
        device=get_device(),
    )

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

    clients: List[LocalBuffClient] = []
    for cid in range(n):
        subset = dd.get_client_data(cid)
        base_delay = per_client_base_delay.get(cid, 0.0)
        is_slow = cid in slow_ids
        clients.append(LocalBuffClient(
            cid=cid,
            cfg=cfg,
            subset=subset,
            work_dir=cfg["io"]["checkpoints_dir"] + "/clients",
            base_delay=base_delay,
            slow=is_slow,
            delay_ranges=((float(a_s), float(b_s)), (float(a_f), float(b_f))),
            jitter=jitter,
            fix_delay=fix_delays,
        ))

    sem = threading.Semaphore(int(cfg["clients"]["concurrent"]))

    def client_loop(client: LocalBuffClient):
        while True:
            with sem:
                cont = client.fit_once(server)
            if not cont:
                break
            time.sleep(0.05)

    server.start_eval_timer()

    threads = []
    for cl in clients:
        t = threading.Thread(target=client_loop, args=(cl,), daemon=False)
        t.start()
        threads.append(t)

    server.wait()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
