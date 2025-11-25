"""TrustWeighted/task.py

CIFAR-10 classifier + data loading utilities with Dirichlet non-IID partitioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
import pytorch_lightning as pl


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent
_CFG_PATH = _ROOT / "config.yml"

if _CFG_PATH.exists():
    with open(_CFG_PATH, "r") as f:
        _RAW_CFG = yaml.safe_load(f) or {}
else:
    _RAW_CFG = {}

_DIR_CFG = _RAW_CFG.get("data", {}) if isinstance(_RAW_CFG, dict) else {}
DIRICHLET_ALPHA: float = float(_DIR_CFG.get("dirichlet_alpha", 0.5))


@dataclass
class DataConfig:
    data_dir: str = "./data"
    batch_size: int = 64
    num_workers: int = 2
    val_fraction: float = 0.1


DATA_CFG = DataConfig()


# ---------------------------------------------------------------------------
# Model definition (SqueezeNet classifier, same as FedAsync)
# ---------------------------------------------------------------------------

def build_squeezenet(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Create SqueezeNet v1.1 and replace the classifier head.

    This mirrors utils/model.py so TrustWeighted uses the same architecture
    as FedAsync/FedBuff.
    """
    if pretrained:
        m = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
    else:
        m = models.squeezenet1_1(weights=None)
    m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1, bias=True)
    m.num_classes = num_classes
    return m


class LitCifar(pl.LightningModule):
    """LightningModule wrapping SqueezeNet for CIFAR-10 classification."""

    def __init__(self, lr: float = 1e-3, num_classes: int = 10):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_squeezenet(num_classes=num_classes, pretrained=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc, prog_bar=False, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ---------------------------------------------------------------------------
# Data loading with Dirichlet non-IID partitioning
# ---------------------------------------------------------------------------

def _cifar_transform():
    # Same normalization as FedAsync client (_testloader in FedAsync/client.py)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _get_full_datasets() -> Tuple[Dataset, Dataset]:
    tfm = _cifar_transform()
    train_ds = datasets.CIFAR10(
        root=DATA_CFG.data_dir, train=True, download=True, transform=tfm
    )
    test_ds = datasets.CIFAR10(
        root=DATA_CFG.data_dir, train=False, download=True, transform=tfm
    )
    return train_ds, test_ds


def _dirichlet_partition(
    labels: Sequence[int],
    num_partitions: int,
    alpha: float,
    seed: int = 42,
) -> List[np.ndarray]:
    """Return indices for each client using class-balanced Dirichlet splits.

    All clients run this locally but we fix the RNG seed so they obtain the
    exact same partition mapping.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    num_classes = int(labels.max()) + 1

    client_indices: List[List[int]] = [[] for _ in range(num_partitions)]

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)

        # Sample proportions for this class among clients
        props = rng.dirichlet(alpha * np.ones(num_partitions))

        # Compute split points and split indices
        split_points = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, split_points)

        for cid, idxs in enumerate(splits):
            client_indices[cid].extend(idxs.tolist())

    # Sort indices for each client for reproducibility
    return [np.array(sorted(ixs), dtype=np.int64) for ixs in client_indices]


def load_data(
    partition_id: int,
    num_partitions: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders for a given client partition."""
    assert 0 <= partition_id < num_partitions, "Invalid partition id"

    train_ds_full, test_ds = _get_full_datasets()
    labels = train_ds_full.targets  # List[int]

    client_splits = _dirichlet_partition(
        labels=labels,
        num_partitions=num_partitions,
        alpha=DIRICHLET_ALPHA,
        seed=42,
    )
    client_idx = client_splits[partition_id]

    # Train/val split inside the client
    rng = np.random.default_rng(1234 + partition_id)
    perm = rng.permutation(len(client_idx))
    client_idx = client_idx[perm]

    val_size = int(len(client_idx) * DATA_CFG.val_fraction)
    val_idx = client_idx[:val_size]
    train_idx = client_idx[val_size:]

    train_ds = Subset(train_ds_full, train_idx.tolist())
    val_ds = Subset(train_ds_full, val_idx.tolist())

    trainloader = DataLoader(
        train_ds,
        batch_size=DATA_CFG.batch_size,
        shuffle=True,
        num_workers=DATA_CFG.num_workers,
        persistent_workers=DATA_CFG.num_workers > 0,
    )

    valloader = DataLoader(
        val_ds,
        batch_size=DATA_CFG.batch_size,
        shuffle=False,
        num_workers=DATA_CFG.num_workers,
        persistent_workers=DATA_CFG.num_workers > 0,
    )

    testloader = DataLoader(
        test_ds,
        batch_size=DATA_CFG.batch_size,
        shuffle=False,
        num_workers=DATA_CFG.num_workers,
        persistent_workers=DATA_CFG.num_workers > 0,
    )

    return trainloader, valloader, testloader


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model(lr: float = 1e-3) -> LitCifar:
    return LitCifar(lr=lr)
