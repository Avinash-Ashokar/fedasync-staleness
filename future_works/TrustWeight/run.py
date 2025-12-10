# Orchestrator: partitions data, starts server, runs async clients
import logging
import random
from typing import List

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torchvision import datasets, transforms

from .config import load_config
from .client import AsyncClient
from .server import AsyncServer
from utils.partitioning import DataDistributor  # <-- NEW: use external partitioner
from utils.helper import get_device 


def _set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_train_dataset(data_dir: str) -> datasets.CIFAR10:
    """(Kept for compatibility; no longer used for partitioning.)"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    return ds


def main() -> None:
    print(f"Using device: {get_device()}")
    logging.basicConfig(level=logging.INFO)

    # --------------------- load config & seed ---------------------
    cfg = load_config()
    _set_seed(cfg.seed)

    # --------------------- dataset & partition --------------------
    # Use DataDistributor as the ONLY mechanism to distribute data.
    # This respects the partitioning behavior defined in utils.partitioning.
    distributor = DataDistributor(
        dataset_name=cfg.data.dataset,   # e.g. "cifar10"
        data_dir=cfg.data.data_dir,
    )
    distributor.distribute_data(
        num_clients=cfg.clients.total,
        alpha=cfg.partition_alpha,
        seed=cfg.seed,
    )

    # Convert the dict {client_id: [indices]} into the list-of-lists expected below
    partitions = [
        distributor.partitions[cid] for cid in range(cfg.clients.total)
    ]

    # --------------------- create server -------------------------
    # AsyncServer encapsulates the trust-weighted async aggregation logic + logging.
    server = AsyncServer(cfg=cfg)

    # --------------------- create clients ------------------------
    clients: List[AsyncClient] = []
    for cid in range(cfg.clients.total):
        indices = partitions[cid] if cid < len(partitions) else []
        client = AsyncClient(
            cid=cid,
            indices=indices,
            cfg=cfg,
        )
        clients.append(client)

    def client_loop(cl: AsyncClient) -> None:
        """Loop for a single client: fetch global, train, send update, repeat."""
        while not server.should_stop():
            cont = cl.run_once(server)
            if not cont or server.should_stop():
                break

    # --------------------- start client threads via executor ------------------
    # Pool size enforces the concurrent-client limit.
    with ThreadPoolExecutor(max_workers=cfg.clients.concurrent) as executor:
        futures = [executor.submit(client_loop, cl) for cl in clients]

        # --------------------- wait for completion -------------------
        # Server decides when to stop (based on target accuracy / max rounds).
        server.wait()

        # Ensure all client loops exit before shutting down.
        for f in futures:
            try:
                f.result(timeout=1.0)
            except Exception:
                pass


if __name__ == "__main__":
    main()
