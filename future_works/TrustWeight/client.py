# Lightning-free async federated client with update compression
import time
import random
from typing import Sequence, Tuple, Dict
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.nn.utils import clip_grad_norm_

from utils.model import build_resnet18
from utils.helper import get_device

from .config import GlobalConfig


def _build_transform() -> transforms.Compose:
    # CIFAR-10 standard augmentation to reduce overfitting
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def _make_dataloader(
    data_dir: str,
    indices: Sequence[int],
    batch_size: int,
) -> DataLoader:
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=_build_transform(),
    )
    subset = Subset(dataset, indices)

    # Safety: in case partitioning ever returns an empty list for a client
    if len(subset) == 0:
        return DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    # num_workers=0 to avoid multiprocessing issues on macOS / Python 3.13
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)


class AsyncClient:
    """Asynchronous client with update compression for communication efficiency.

    Improvements:
    - Delta compression: sends only the difference from base model
    - Quantization: reduces update precision to save bandwidth
    - Sparsification: sends only top-k significant updates
    """

    def __init__(
        self,
        cid: int,
        indices: Sequence[int],
        cfg: GlobalConfig,
    ) -> None:
        self.cid = cid
        self.cfg = cfg
        self.device = get_device()
        self.loader = _make_dataloader(
            cfg.data.data_dir,
            indices,
            cfg.clients.batch_size,
        )
        self.num_classes = cfg.data.num_classes
        self.local_epochs = cfg.clients.local_epochs
        self.lr = cfg.clients.lr
        self.weight_decay = cfg.clients.weight_decay
        self.grad_clip = cfg.clients.grad_clip

        # Compression settings
        self.use_compression = getattr(cfg, 'use_compression', True)
        self.compression_ratio = getattr(cfg, 'compression_ratio', 0.5)  # 50% compression

        # --- straggler behaviour ---
        num_clients = cfg.clients.total
        slow_fraction = cfg.clients.struggle_percent / 100.0
        num_slow = int(round(num_clients * slow_fraction))
        self.is_slow = cid < num_slow  # deterministic but good enough

        self.delay_slow_range = tuple(cfg.clients.delay_slow_range)
        self.delay_fast_range = tuple(cfg.clients.delay_fast_range)
        self.jitter_per_round = float(cfg.clients.jitter_per_round)
        self.client_delay = float(cfg.server_runtime.client_delay)
        print(cid)

    # ------------------------------------------------------------------ utils

    def _sample_delay(self) -> float:
        if self.is_slow:
            base = random.uniform(*self.delay_slow_range)
        else:
            base = random.uniform(*self.delay_fast_range)
        jitter = random.uniform(-self.jitter_per_round, self.jitter_per_round)
        return max(0.0, base + jitter + self.client_delay)

    def _build_model(self) -> nn.Module:
        model = build_resnet18(num_classes=self.num_classes)
        return model.to(self.device)

    def _compress_update(
        self, 
        update_vec: torch.Tensor, 
        base_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """Compress update using delta compression, quantization, and sparsification.
        
        Args:
            update_vec: New model parameters (flattened)
            base_vec: Base model parameters (flattened)
        
        Returns:
            Compressed update and metadata
        """
        # 1. Delta compression: compute difference
        delta = update_vec - base_vec
        
        # 2. Quantization: reduce to 8-bit
        delta_min = delta.min()
        delta_max = delta.max()
        scale = (delta_max - delta_min) / 255.0 if (delta_max - delta_min) > 1e-8 else 1.0
        delta_quantized = torch.round((delta - delta_min) / scale).clamp(0, 255).byte()
        
        # 3. Sparsification: keep only top-k values
        k = int(len(delta) * self.compression_ratio)
        if k < len(delta):
            _, top_indices = torch.topk(torch.abs(delta), k)
            # Create sparse representation
            delta_sparse_values = delta[top_indices]
            delta_sparse_indices = top_indices
        else:
            delta_sparse_values = delta
            delta_sparse_indices = torch.arange(len(delta))
        
        metadata = {
            'min': delta_min.item(),
            'scale': scale.item(),
            'indices': delta_sparse_indices.cpu(),
            'values': delta_sparse_values.cpu(),
            'original_shape': delta.shape,
            'compressed': True
        }
        
        return delta_quantized, metadata

    # ---------------------------------------------------------------- training

    def _evaluate_on_loader(self, model: nn.Module) -> Tuple[float, float]:
        """Return (loss, accuracy) on the client's local data."""
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                total_correct += (preds == yb).sum().item()
                total_examples += xb.size(0)
        if total_examples == 0:
            return 0.0, 0.0
        return total_loss / total_examples, total_correct / total_examples

    def _train_local(self, model: nn.Module) -> Tuple[float, float]:
        """Train for `local_epochs` and return (loss_after, acc_after)."""
        model.train()
        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )

        for _ in range(self.local_epochs):
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optim.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                if self.grad_clip > 0:
                    clip_grad_norm_(model.parameters(), self.grad_clip)
                optim.step()

        # reuse evaluation code for final metrics
        return self._evaluate_on_loader(model)

    # ------------------------------------------------------------- main loop

    def run_once(self, server) -> bool:
        """Perform a single async round with the server, using compression if enabled.

        Returns False when the server indicates global stopping, True otherwise.
        """
        # Simulated network / computation delay heterogeneity
        delay = self._sample_delay()
        if delay > 0:
            time.sleep(delay)

        if server.should_stop():
            return False

        # Get the latest global model snapshot
        version, global_state = server.get_global_model()
        model = self._build_model()
        model.load_state_dict(global_state)
        
        # Store base state for delta compression
        base_state_dict = OrderedDict()
        for k, v in model.state_dict().items():
            base_state_dict[k] = v.detach().cpu().clone()

        # Evaluate before local training to compute loss drop ΔL̃_i
        loss_before, _ = self._evaluate_on_loader(model)

        start_time = time.time()
        loss_after, train_acc = self._train_local(model)
        train_time_s = time.time() - start_time

        # Local "test" is just another pass over the client's data
        test_loss, test_acc = self._evaluate_on_loader(model)

        # Prepare update with compression
        new_params = OrderedDict()
        for k, v in model.state_dict().items():
            new_params[k] = v.detach().cpu().clone()
        num_examples = len(self.loader.dataset)

        delta_loss = loss_before - loss_after  # ΔL̃_i

        # All local metrics are computed here and passed to the server
        server.submit_update(
            client_id=self.cid,
            base_version=version,
            new_params=new_params,
            num_samples=num_examples,
            train_time_s=train_time_s,
            delta_loss=delta_loss,
            loss_before=loss_before,
            loss_after=loss_after,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            use_compression=self.use_compression,
            base_state=base_state_dict if self.use_compression else None,
        )
        return not server.should_stop()
