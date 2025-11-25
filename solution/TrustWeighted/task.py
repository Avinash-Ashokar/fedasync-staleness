"""TrustWeighted/task.py

CIFAR-10 model + data loading utilities with Dirichlet non-IID partitioning.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from torchvision import transforms, datasets

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner


# ---------------------------------------------------------------------------
# Load config.yml
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load YAML config containing Dirichlet alpha."""
    with open("TrustWeighted/config.yml", "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()
DIRICHLET_ALPHA = float(CONFIG["data"]["dirichlet_alpha"])


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class LitAutoEncoder(pl.LightningModule):
    """Simple convolutional autoencoder for CIFAR-10."""

    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 16×16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 8×8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 4×4
            nn.ReLU(),
            nn.Flatten(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8×8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16×16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 32×32
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("test_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class HFImageDataset(Dataset):
    """Wrap HuggingFace-like dataset objects into PyTorch Dataset."""
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[int(idx)]
        img = item["img"]
        label = int(item["label"])
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Data Loading with Dirichlet Non-IID
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    batch_size: int = 32
    num_workers: int = 4
    test_num_workers: int = 4
    val_fraction: float = 0.1
    seed: int = 42

DATA_CFG = DataConfig()

def _build_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])


def load_data(partition_id: int, num_partitions: int):
    """Load CIFAR-10 data using Dirichlet non-IID partitioning."""

    fds = FederatedDataset(
        dataset="cifar10",
        partitioners={
            "train": DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label", 
                alpha=DIRICHLET_ALPHA,  # Loaded from config.yml
            )
        },
    )

    # non-IID client partition
    partition = fds.load_partition(partition_id, split="train")

    # Local split: train/val
    partition_train_valid = partition.train_test_split(
        test_size=DATA_CFG.val_fraction, seed=DATA_CFG.seed
    )

    # Centralized test set
    full_test = fds.load_split("test")

    transform = _build_transform()

    train_ds = HFImageDataset(partition_train_valid["train"], transform)
    val_ds = HFImageDataset(partition_train_valid["test"], transform)
    test_ds = HFImageDataset(full_test, transform)

    # DataLoaders
    trainloader = DataLoader(
        train_ds, batch_size=DATA_CFG.batch_size, shuffle=True,
        num_workers=DATA_CFG.num_workers, persistent_workers=True
    )
    valloader = DataLoader(
        val_ds, batch_size=DATA_CFG.batch_size, shuffle=False,
        num_workers=DATA_CFG.num_workers, persistent_workers=True
    )
    testloader = DataLoader(
        test_ds, batch_size=DATA_CFG.batch_size, shuffle=False,
        num_workers=DATA_CFG.test_num_workers, persistent_workers=True
    )

    return trainloader, valloader, testloader


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model(lr=1e-3):
    return LitAutoEncoder(lr=lr)
