import os
import logging
import warnings
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
from datetime import datetime
from pathlib import Path
import subprocess
import shutil

import yaml

from .client import AsyncClient
from .server import AsyncServer
from utils.model import build_resnet18
from utils.partitioning import DataDistributor
from utils.helper import set_seed, get_device


CFG_PATH = os.environ.get("TRUSTWEIGHT_CONFIG", os.path.join(os.path.dirname(__file__), "config.yaml"))


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_cfg(CFG_PATH)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    random.seed(seed)

    run_dir = Path("logs") / "avinash" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except:
        commit_hash = "unknown"
    
    csv_header = "total_agg,avg_train_loss,avg_train_acc,test_loss,test_acc,time"
    with (run_dir / "COMMIT.txt").open("w") as f:
        f.write(f"{commit_hash},{csv_header}\n")
    
    shutil.copy(CFG_PATH, run_dir / "CONFIG.yaml")

    dd = DataDistributor(dataset_name=cfg["data"]["dataset"], data_dir=cfg["data"]["data_dir"])
    dd.distribute_data(
        num_clients=int(cfg["clients"]["total"]),
        alpha=float(cfg.get("partition_alpha", 0.5)),
        seed=seed,
    )

    global_model = build_resnet18(num_classes=cfg["data"]["num_classes"], pretrained=False)
    server = AsyncServer(
        global_model=global_model,
        total_train_samples=len(dd.train_dataset),
        buffer_size=int(cfg["trustweight"]["buffer_size"]),
        buffer_timeout_s=float(cfg["trustweight"]["buffer_timeout_s"]),
        use_sample_weighing=bool(cfg["trustweight"]["use_sample_weighing"]),
        target_accuracy=float(cfg["eval"]["target_accuracy"]),
        max_rounds=int(cfg["train"]["max_rounds"]) if "max_rounds" in cfg["train"] else None,
        eval_interval_s=int(cfg["eval"]["interval_seconds"]),
        data_dir=cfg["data"]["data_dir"],
        checkpoints_dir=str(run_dir / "checkpoints"),
        logs_dir=str(run_dir),
        global_log_csv=str(run_dir / "TrustWeight.csv"),
        client_participation_csv=str(run_dir / "TrustWeightClientParticipation.csv"),
        final_model_path=str(run_dir / "TrustWeightModel.pt"),
        resume=False,
        device=get_device(),
        eta=float(cfg["trustweight"].get("eta", 0.5)),
        theta=tuple(cfg["trustweight"].get("theta", [1.0, -0.1, 0.2])),
        freshness_alpha=float(cfg["trustweight"].get("freshness_alpha", 0.1)),
        beta1=float(cfg["trustweight"].get("beta1", 0.0)),
        beta2=float(cfg["trustweight"].get("beta2", 0.0)),
        momentum_gamma=float(cfg["trustweight"].get("momentum_gamma", 0.9)),
        update_clip_norm=float(cfg["train"].get("update_clip_norm", 5.0)),
    )

    n = int(cfg["clients"]["total"])
    clients: List[AsyncClient] = []
    for cid in range(n):
        indices = dd.partitions[cid] if cid in dd.partitions else []
        clients.append(
            AsyncClient(
                cid=cid,
                indices=indices,
                cfg=cfg,
            )
        )

    sem = threading.Semaphore(int(cfg["clients"]["concurrent"]))

    def client_loop(cl: AsyncClient) -> None:
        while not server.should_stop():
            with sem:
                cont = cl.run_once(server)
            if not cont or server.should_stop():
                break
            time.sleep(0.05)

    threads: List[threading.Thread] = []
    for cl in clients:
        t = threading.Thread(target=client_loop, args=(cl,), daemon=True)
        t.start()
        threads.append(t)

    print(f"[TrustWeight] Started {len(threads)} client threads")
    print(f"[TrustWeight] Running for up to {cfg['train']['max_rounds']} rounds...")
    print(f"[TrustWeight] Results: {run_dir}")

    server.wait()
    for t in threads:
        t.join()

    print(f"[TrustWeight] Training completed. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
