import time
import random
from typing import Sequence, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.nn.utils import clip_grad_norm_

from utils.model import build_resnet18
from utils.helper import get_device


def _build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


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
    
    if len(subset) == 0:
        return DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)


class AsyncClient:
    def __init__(
        self,
        cid: int,
        indices: Sequence[int],
        cfg: Dict,
    ) -> None:
        self.cid = cid
        self.cfg = cfg
        self.device = get_device()
        self.loader = _make_dataloader(
            cfg["data"]["data_dir"],
            indices,
            cfg["clients"]["batch_size"],
        )
        self.num_classes = cfg["data"]["num_classes"]
        self.local_epochs = cfg["clients"]["local_epochs"]
        self.lr = cfg["clients"]["lr"]
        self.weight_decay = cfg["clients"]["weight_decay"]
        self.grad_clip = cfg["clients"]["grad_clip"]
        
        num_clients = cfg["clients"]["total"]
        slow_fraction = cfg["clients"].get("struggle_percent", 0) / 100.0
        num_slow = int(round(num_clients * slow_fraction))
        self.is_slow = cid < num_slow
        
        self.delay_slow_range = tuple(cfg["clients"].get("delay_slow_range", [0.8, 2.0]))
        self.delay_fast_range = tuple(cfg["clients"].get("delay_fast_range", [0.0, 0.2]))
        self.jitter_per_round = float(cfg["clients"].get("jitter_per_round", 0.0))
        self.client_delay = float(cfg.get("server_runtime", {}).get("client_delay", 0.0))
    
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
    
    def _evaluate_on_loader(self, model: nn.Module) -> Tuple[float, float]:
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
        model.train()
        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.cfg["clients"].get("momentum", 0.9),
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
        
        return self._evaluate_on_loader(model)
    
    def run_once(self, server) -> bool:
        delay = self._sample_delay()
        if delay > 0:
            time.sleep(delay)
        
        if server.should_stop():
            return False
        
        version, global_state = server.get_global_model()
        model = self._build_model()
        model.load_state_dict(global_state)
        
        loss_before, _ = self._evaluate_on_loader(model)
        
        start_time = time.time()
        loss_after, train_acc = self._train_local(model)
        train_time_s = time.time() - start_time
        
        test_loss, test_acc = self._evaluate_on_loader(model)
        
        from collections import OrderedDict
        new_params = OrderedDict()
        for k, v in model.state_dict().items():
            new_params[k] = v.detach().cpu().clone()
        num_examples = len(self.loader.dataset)
        
        delta_loss = loss_before - loss_after
        
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
        )
        return not server.should_stop()
