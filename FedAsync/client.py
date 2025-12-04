# Lightning-based local client, resumable checkpoints, no Flower deps
import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import pytorch_lightning as pl

from utils.model import build_resnet18, state_to_list, list_to_state
from utils.helper import get_device
import random


def _device_to_accelerator(device: torch.device) -> str:
    if device.type == "cuda":
        return "gpu"
    if device.type == "mps":
        return "mps"
    return "cpu"


def _testloader(root: str, batch_size: int = 256):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    crit = nn.CrossEntropyLoss()
    model = model.to(device)
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            loss_sum += float(loss.item()) * y.size(0)
            total += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    return loss_sum / max(1, total), correct / max(1, total)


class LitCifar(pl.LightningModule):
    def __init__(self, base_model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["base_model"])
        self.model = base_model
        self.criterion = nn.CrossEntropyLoss()
        self._train_loss_sum = 0.0
        self._train_correct = 0
        self._train_total = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        pred = logits.argmax(1)
        self._train_loss_sum += float(loss.item()) * y.size(0)
        self._train_correct += (pred == y).sum().item()
        self._train_total += y.size(0)
        return loss

    def on_train_epoch_start(self):
        self._train_loss_sum = 0.0
        self._train_correct = 0
        self._train_total = 0

    def get_epoch_metrics(self) -> Tuple[float, float]:
        if self._train_total == 0:
            return 0.0, 0.0
        return self._train_loss_sum / self._train_total, self._train_correct / self._train_total

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class LocalAsyncClient:
    """Pull global -> Lightning local fit -> push update with averaged metrics.
       Supports per-client slow/fast delays with optional per-round jitter."""
    def __init__(
        self,
        cid: int,
        cfg: dict,
        subset: Subset,
        work_dir: str = "./checkpoints/clients",
        base_delay: float = 0.0,
        slow: bool = False,
        delay_ranges: Optional[tuple] = None,   # ((a_s, b_s), (a_f, b_f))
        jitter: float = 0.0,
        fix_delay: bool = True,
    ):
        self.cid = cid
        self.cfg = cfg
        self.device = get_device()

        base = build_resnet18(num_classes=cfg["data"]["num_classes"], pretrained=False)
        self.lit = LitCifar(base, lr=float(cfg["clients"]["lr"]))

        self.loader = DataLoader(subset, batch_size=int(cfg["clients"]["batch_size"]),
                                 shuffle=True, num_workers=0)

        self.client_dir = Path(work_dir) / f"cid_{cid}"
        self.client_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = str(self.client_dir / "last.ckpt")

        # delay controls
        self.base_delay = float(base_delay)
        self.slow = bool(slow)
        self.delay_ranges = delay_ranges
        self.jitter = float(jitter)
        self.fix_delay = bool(fix_delay)

        # pre-sample fixed delay if requested
        if self.fix_delay and self.delay_ranges is not None:
            (a_s, b_s), (a_f, b_f) = self.delay_ranges
            if self.slow:
                self.base_delay = random.uniform(float(a_s), float(b_s))
            else:
                self.base_delay = random.uniform(float(a_f), float(b_f))

        self.accelerator = _device_to_accelerator(self.device)

        # local test loader to compute per-client test metrics
        self.testloader = _testloader(cfg["data"]["data_dir"])

    def _to_list(self) -> List[torch.Tensor]:
        return state_to_list(self.lit.model.state_dict())

    def _from_list(self, arrs: List[torch.Tensor]) -> None:
        sd = self.lit.model.state_dict()
        new_sd = list_to_state(sd, arrs)
        self.lit.model.load_state_dict(new_sd, strict=True)
        self.lit.to(self.device)

    def _sleep_delay(self):
        # global delay from config (kept for backward compat)
        global_d = float(self.cfg.get("server_runtime", {}).get("client_delay", 0.0))

        # per-client base delay
        base = self.base_delay

        # if not fixed, resample each fit
        if not self.fix_delay and self.delay_ranges is not None:
            (a_s, b_s), (a_f, b_f) = self.delay_ranges
            if self.slow:
                base = random.uniform(float(a_s), float(b_s))
            else:
                base = random.uniform(float(a_f), float(b_f))

        # add +/- jitter
        jit = random.uniform(-self.jitter, self.jitter) if self.jitter > 0.0 else 0.0

        delay = max(0.0, global_d + base + jit)
        if delay > 0.0:
            time.sleep(delay)

    def fit_once(self, server) -> bool:
        # pull global
        params, version = server.get_global()
        self._from_list(params)

        # emulate heterogeneous device speed
        self._sleep_delay()

        # train for local_epochs; checkpoints disabled for async runs
        epochs = int(self.cfg["clients"]["local_epochs"])
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=self.accelerator,
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            callbacks=[],
        )
        start = time.time()
        # ckpt = self.ckpt_path if Path(self.ckpt_path).exists() else None
        trainer.fit(self.lit, train_dataloaders=self.loader)
        duration = time.time() - start

        # local metrics
        train_loss, train_acc = self.lit.get_epoch_metrics()
        test_loss, test_acc = _evaluate(self.lit.model, self.testloader, self.device)

        new_params = self._to_list()
        num_examples = len(self.loader.dataset)

        server.submit_update(
            client_id=self.cid,
            base_version=version,
            new_params=new_params,
            num_samples=num_examples,
            train_time_s=duration,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
        )
        return not server.should_stop()
