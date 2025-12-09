import csv
import time
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, OrderedDict as ODType
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.model import build_resnet18
from utils.helper import get_device
from .strategy import TrustWeightedAsyncStrategy, TrustWeightedConfig


def _testloader(root: str, batch_size: int = 256) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _flatten_state(state: ODType[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in state.values()])


def _flatten_state_by_template(
    state: Dict[str, torch.Tensor], template: ODType[str, torch.Tensor]
) -> torch.Tensor:
    return torch.cat([state[k].reshape(-1) for k in template.keys()])


def _vector_to_state(
    vec: torch.Tensor, template: ODType[str, torch.Tensor]
) -> ODType[str, torch.Tensor]:
    new_state: ODType[str, torch.Tensor] = type(template)()
    offset = 0
    for k, t in template.items():
        numel = t.numel()
        new_state[k] = vec[offset : offset + numel].view_as(t).clone()
        offset += numel
    assert offset == vec.numel()
    return new_state


@dataclass
class ClientUpdateState:
    client_id: int
    base_version: int
    new_params: ODType[str, torch.Tensor]
    num_samples: int
    train_time_s: float
    delta_loss: float
    loss_before: float
    loss_after: float
    train_acc: float
    test_loss: float
    test_acc: float
    arrival_ts: float


class AsyncServer:
    def __init__(
        self,
        global_model: torch.nn.Module,
        total_train_samples: int,
        buffer_size: int = 5,
        buffer_timeout_s: float = 5.0,
        use_sample_weighing: bool = True,
        target_accuracy: float = 0.8,
        max_rounds: Optional[int] = None,
        eval_interval_s: int = 15,
        data_dir: str = "./data",
        checkpoints_dir: str = "./checkpoints/TrustWeight",
        logs_dir: str = "./logs/TrustWeight",
        global_log_csv: Optional[str] = None,
        client_participation_csv: Optional[str] = None,
        final_model_path: Optional[str] = None,
        resume: bool = True,
        device: Optional[torch.device] = None,
        eta: float = 1.0,
        theta: Optional[Tuple[float, float, float]] = None,
        freshness_alpha: float = 0.1,
        beta1: float = 0.0,
        beta2: float = 0.0,
        momentum_gamma: float = 0.9,
        update_clip_norm: float = 5.0,
    ):
        self.device = device or get_device()
        
        if hasattr(global_model, 'num_classes'):
            self.num_classes = global_model.num_classes
        elif hasattr(global_model, 'fc'):
            self.num_classes = global_model.fc.out_features
        else:
            self.num_classes = 10
        
        self.testloader = _testloader(data_dir, batch_size=256)
        
        self._template_state: ODType[str, torch.Tensor] = ODType(
            (k, v.detach().cpu().clone()) for k, v in global_model.state_dict().items()
        )
        self._global_state: ODType[str, torch.Tensor] = ODType(
            (k, v.clone()) for k, v in self._template_state.items()
        )
        
        self._model_versions: List[ODType[str, torch.Tensor]] = [
            ODType((k, v.clone()) for k, v in self._global_state.items())
        ]
        self._version: int = 0
        
        dim = _flatten_state(self._global_state).numel()
        theta_cfg = theta if theta is not None else (1.0, -0.1, 0.2)
        strategy_cfg = TrustWeightedConfig(
            eta=eta,
            theta=theta_cfg,
            freshness_alpha=freshness_alpha,
            beta1=beta1,
            beta2=beta2,
            momentum_gamma=momentum_gamma,
        )
        self.strategy = TrustWeightedAsyncStrategy(dim=dim, cfg=strategy_cfg)
        self.eta = eta
        self.theta = theta_cfg
        
        self.buffer: List[ClientUpdateState] = []
        self.buffer_size: int = buffer_size
        self.buffer_timeout_s: float = buffer_timeout_s
        self._last_flush_ts: float = time.time()
        
        self.global_log_path = Path(global_log_csv) if global_log_csv else Path(logs_dir) / "TrustWeight.csv"
        self.global_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_global_log()
        
        self.client_log_path = Path(client_participation_csv) if client_participation_csv else Path(logs_dir) / "TrustWeightClientParticipation.csv"
        self.client_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_client_log()
        
        self.eval_interval_s: float = eval_interval_s
        self.target_accuracy: float = target_accuracy
        self.max_rounds: int = max_rounds if max_rounds is not None else 1000
        self.update_clip_norm: float = update_clip_norm
        
        self._num_aggregations: int = 0
        self._stop: bool = False
        self._stop_reason: str = ""
        self._lock = threading.Lock()
        self._agg_lock = threading.Lock()
        self._start_ts: float = time.time()
        
        self._eval_thread = None
    
    def _init_global_log(self) -> None:
        if self.global_log_path.exists():
            return
        with self.global_log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "total_agg", "avg_train_loss", "avg_train_acc",
                "test_loss", "test_acc", "time",
            ])
    
    def _append_global_log(
        self,
        total_agg: int,
        avg_train_loss: float,
        avg_train_acc: float,
        test_loss: float,
        test_acc: float,
    ) -> None:
        ts = time.time() - self._start_ts
        with self.global_log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                int(total_agg),
                float(avg_train_loss),
                float(avg_train_acc),
                float(test_loss),
                float(test_acc),
                ts,
            ])
    
    def _init_client_log(self) -> None:
        if self.client_log_path.exists():
            return
        with self.client_log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "client_id", "local_train_loss", "local_train_acc",
                "local_test_loss", "local_test_acc", "total_agg", "staleness",
            ])
    
    def _append_client_participation_log(
        self,
        total_agg: int,
        updates: List[ClientUpdateState],
        staleness_list: List[float],
    ) -> None:
        with self.client_log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            for u, tau_i in zip(updates, staleness_list):
                writer.writerow([
                    int(u.client_id),
                    float(u.loss_after),
                    float(u.train_acc),
                    float(u.test_loss),
                    float(u.test_acc),
                    int(total_agg),
                    float(tau_i),
                ])
    
    def should_stop(self) -> bool:
        with self._lock:
            return self._stop
    
    def mark_stop(self, reason: str = "") -> None:
        with self._lock:
            if not self._stop:
                self._stop = True
                if reason:
                    self._stop_reason = reason
                print(f"[Server] Stopping: {self._stop_reason}")
    
    def get_global_model(self) -> Tuple[int, Dict[str, torch.Tensor]]:
        with self._lock:
            version = self._version
            state = ODType((k, v.clone()) for k, v in self._global_state.items())
        return version, state
    
    def _make_model_from_state(self, state: Dict[str, torch.Tensor], num_classes: int) -> torch.nn.Module:
        model = build_resnet18(num_classes=num_classes)
        model.load_state_dict(state)
        return model.to(self.device)
    
    def _evaluate_global(self, num_classes: int) -> Tuple[float, float]:
        model = self._make_model_from_state(self._global_state, num_classes)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for xb, yb in self.testloader:
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
    
    def _flush_buffer_if_needed(self) -> None:
        now = time.time()
        should_flush = False
        if len(self.buffer) >= self.buffer_size:
            should_flush = True
        elif (now - self._last_flush_ts) >= self.buffer_timeout_s and self.buffer:
            should_flush = True
        
        if not should_flush:
            return
        
        with self._lock:
            buffer_copy = list(self.buffer)
            self.buffer.clear()
            self._last_flush_ts = now
        
        with self._agg_lock:
            self._aggregate(buffer_copy)
    
    def _aggregate(self, updates: List[ClientUpdateState]) -> None:
        if not updates:
            return
        
        with self._lock:
            global_vec = _flatten_state_by_template(self._global_state, self._template_state)
            version_now = self._version
            model_versions = list(self._model_versions)
        
        update_vectors: List[Dict[str, torch.Tensor]] = []
        staleness_list: List[float] = []
        valid_updates: List[ClientUpdateState] = []
        
        for u in updates:
            base_state = model_versions[u.base_version]
            base_vec = _flatten_state_by_template(base_state, self._template_state)
            new_vec = _flatten_state_by_template(u.new_params, self._template_state)
            ui = new_vec - base_vec
            
            if not torch.isfinite(ui).all():
                print(f"[Server] Dropping client {u.client_id} update due to NaN/Inf values")
                continue
            if self.update_clip_norm > 0:
                norm = torch.norm(ui)
                if torch.isfinite(norm) and norm.item() > self.update_clip_norm:
                    ui = ui * (self.update_clip_norm / (norm + 1e-12))
            
            tau_i = float(max(0, version_now - u.base_version))
            staleness_list.append(tau_i)
            
            delta_loss = float(u.delta_loss)
            update_vectors.append({
                "u": ui,
                "tau": torch.tensor(tau_i, dtype=torch.float32),
                "num_samples": torch.tensor(float(u.num_samples), dtype=torch.float32),
                "delta_loss": torch.tensor(delta_loss, dtype=torch.float32),
            })
            valid_updates.append(u)
        
        if not update_vectors:
            print("[Server] Buffer flush skipped: no valid updates after filtering.")
            return
        
        new_global_vec, agg_metrics = self.strategy.aggregate(global_vec, update_vectors)
        
        new_state = _vector_to_state(new_global_vec, self._template_state)
        
        avg_train_loss = sum(u.loss_after for u in valid_updates) / len(valid_updates)
        avg_train_acc = sum(u.train_acc for u in valid_updates) / len(valid_updates)
        
        with self._lock:
            self._global_state = ODType((k, v.clone()) for k, v in new_state.items())
            self._model_versions.append(
                ODType((k, v.clone()) for k, v in self._global_state.items())
            )
            self._version = len(self._model_versions) - 1
            self._num_aggregations += 1
            total_agg = self._num_aggregations
        
        test_loss, test_acc = self._evaluate_global(self.num_classes)
        
        self._append_global_log(
            total_agg=total_agg,
            avg_train_loss=avg_train_loss,
            avg_train_acc=avg_train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
        )
        self._append_client_participation_log(
            total_agg=total_agg,
            updates=valid_updates,
            staleness_list=staleness_list,
        )
        
        print(
            f"[Server] Aggregated {len(valid_updates)} updates -> agg={total_agg} "
            f"(avg_tau={agg_metrics.get('avg_tau', 0.0):.3f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f})"
        )
        
        if test_acc >= self.target_accuracy:
            self.mark_stop(f"target accuracy {test_acc:.4f} reached")
        if total_agg >= self.max_rounds:
            self.mark_stop("max aggregation rounds reached")
    
    def submit_update(
        self,
        client_id: int,
        base_version: int,
        new_params: Dict[str, torch.Tensor],
        num_samples: int,
        train_time_s: float,
        delta_loss: float,
        loss_before: float,
        loss_after: float,
        train_acc: float,
        test_loss: float,
        test_acc: float,
    ) -> None:
        cu = ClientUpdateState(
            client_id=client_id,
            base_version=base_version,
            new_params=new_params,
            num_samples=num_samples,
            train_time_s=float(train_time_s),
            delta_loss=float(delta_loss),
            loss_before=float(loss_before),
            loss_after=float(loss_after),
            train_acc=float(train_acc),
            test_loss=float(test_loss),
            test_acc=float(test_acc),
            arrival_ts=time.time(),
        )
        with self._lock:
            self.buffer.append(cu)
        self._flush_buffer_if_needed()
    
    def start_eval_timer(self):
        def _loop():
            next_ts = time.time() + self.eval_interval_s
            while True:
                now = time.time()
                sleep_for = max(0.0, next_ts - now)
                time.sleep(sleep_for)
                with self._lock:
                    if self._stop:
                        break
                next_ts += self.eval_interval_s
        self._eval_thread = threading.Thread(target=_loop, daemon=True)
        self._eval_thread.start()
    
    def wait(self) -> None:
        try:
            while not self.should_stop():
                time.sleep(0.2)
        finally:
            self.mark_stop(self._stop_reason or "training finished")
