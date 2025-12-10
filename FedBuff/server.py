import csv
import time
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.model import state_to_list, list_to_state
from utils.helper import get_device


def _testloader(root: str, batch_size: int = 256):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += float(loss.item()) * y.size(0)
            total += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    return loss_sum / max(1, total), correct / max(1, total)


class BufferedFedServer:
    def __init__(
        self,
        global_model: torch.nn.Module,
        total_train_samples: int,
        buffer_size: int = 5,
        buffer_timeout_s: float = 10.0,
        use_sample_weighing: bool = True,
        target_accuracy: float = 0.70,
        max_rounds: Optional[int] = None,
        eval_interval_s: int = 15,
        data_dir: str = "./data",
        checkpoints_dir: str = "./checkpoints",
        logs_dir: str = "./logs",
        global_log_csv: Optional[str] = None,
        client_participation_csv: Optional[str] = None,
        final_model_path: Optional[str] = None,
        resume: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.model = global_model
        self.template = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.device = device or get_device()
        self.model.to(self.device)

        self.total_train_samples = int(total_train_samples)
        self.buffer_size = int(buffer_size)
        self.buffer_timeout_s = float(buffer_timeout_s)
        self.use_sample_weighing = bool(use_sample_weighing)

        self.eval_interval_s = int(eval_interval_s)
        self.target_accuracy = float(target_accuracy)
        self.max_rounds = int(max_rounds) if max_rounds is not None else None

        self.data_dir = data_dir
        self.ckpt_dir = Path(checkpoints_dir); self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(logs_dir); self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = Path(global_log_csv) if global_log_csv else (self.log_dir / "FedBuff.csv")
        self.participation_csv = Path(client_participation_csv) if client_participation_csv else (self.log_dir / "FedBuffClientParticipation.csv")
        self.final_model_path = Path(final_model_path) if final_model_path else Path("./results/FedBuffModel.pt")
        self.final_model_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(["total_agg", "avg_train_loss", "avg_train_acc",
                                        "test_loss", "test_acc", "time"])

        if not self.participation_csv.exists():
            self.participation_csv.parent.mkdir(parents=True, exist_ok=True)
            with self.participation_csv.open("w", newline="") as f:
                csv.writer(f).writerow([
                    "client_id", "local_train_loss", "local_train_acc",
                    "local_test_loss", "local_test_acc", "total_agg"
                ])

        self._lock = threading.Lock()
        self._stop = False
        self.t_round = 0
        self._log_count = 0
        self.testloader = _testloader(self.data_dir)
        self._train_loss_acc_accum: List[Tuple[float, float, int]] = []
        self._start_ts = time.time()

        self._buffer: List[Dict] = []
        self._buffer_last_flush = time.time()

        if resume:
            self._maybe_resume()

    def _ckpt_file(self) -> Path:
        return self.ckpt_dir / "server_last.ckpt"

    def _maybe_resume(self) -> None:
        ck = self._ckpt_file()
        if ck.exists():
            blob = torch.load(ck, map_location="cpu")
            state = list_to_state(self.template, blob["global_params"])
            self.model.load_state_dict(state, strict=True)
            self.t_round = int(blob["t_round"])
            print(f"[resume] Loaded server checkpoint at total_agg={self.t_round}")

    def _save_ckpt(self) -> None:
        sd = state_to_list(self.model.state_dict())
        torch.save({"t_round": self.t_round, "global_params": sd}, self._ckpt_file())

    def _save_final_model(self) -> None:
        torch.save(self.model.state_dict(), self.final_model_path)

    def get_global(self):
        with self._lock:
            return state_to_list(self.model.state_dict()), self.t_round

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return

        g = state_to_list(self.model.state_dict())
        total_samples = sum(u["num_samples"] for u in self._buffer)

        merged = [torch.zeros_like(gi, device=gi.device) for gi in g]
        for u in self._buffer:
            weight = float(u["num_samples"]) / float(total_samples) if self.use_sample_weighing else 1.0 / len(self._buffer)
            for i, ci in enumerate(u["new_params"]):
                ci_tensor = ci.to(merged[i].device).type_as(merged[i])
                merged[i] += weight * ci_tensor

        new_state = list_to_state(self.template, merged)
        self.model.load_state_dict(new_state, strict=True)

        for u in self._buffer:
            self._train_loss_acc_accum.append((u["train_loss"], u["train_acc"], u["num_samples"]))

        self._buffer.clear()
        self._buffer_last_flush = time.time()
        self.t_round += 1
        self._save_ckpt()

    def submit_update(
        self,
        client_id: int,
        base_version: int,
        new_params: List[torch.Tensor],
        num_samples: int,
        train_time_s: float,
        train_loss: float,
        train_acc: float,
        test_loss: float,
        test_acc: float,
    ) -> None:
        with self._lock:
            if self._stop:
                return
            if self.max_rounds is not None and self.t_round >= self.max_rounds:
                self._stop = True
                return

            with self.participation_csv.open("a", newline="") as f:
                csv.writer(f).writerow([
                    client_id, f"{train_loss:.6f}", f"{train_acc:.6f}",
                    f"{test_loss:.6f}", f"{test_acc:.6f}", self.t_round
                ])

            self._buffer.append({
                "client_id": client_id,
                "base_version": base_version,
                "new_params": new_params,
                "num_samples": num_samples,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
            })

            should_flush = False
            if len(self._buffer) >= self.buffer_size:
                should_flush = True
            elif time.time() - self._buffer_last_flush >= self.buffer_timeout_s:
                should_flush = True

            if should_flush:
                self._flush_buffer()

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop

    def mark_stop(self) -> None:
        with self._lock:
            if self._buffer:
                self._flush_buffer()
            self._stop = True
            self._save_final_model()
            print(f"[LOG] saved final model -> {self.final_model_path}")

    def _compute_avg_train(self) -> Tuple[float, float]:
        if not self._train_loss_acc_accum:
            return 0.0, 0.0
        loss_sum, acc_sum, n_sum = 0.0, 0.0, 0
        for l, a, n in self._train_loss_acc_accum:
            loss_sum += l * n
            acc_sum += a * n
            n_sum += n
        return loss_sum / max(1, n_sum), acc_sum / max(1, n_sum)

    def _periodic_eval_and_log(self):
        test_loss, test_acc = _evaluate(self.model, self.testloader, self.device)
        avg_train_loss, avg_train_acc = self._compute_avg_train()
        now = time.time() - self._start_ts
        self._train_loss_acc_accum.clear()

        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                self.t_round, f"{avg_train_loss:.6f}", f"{avg_train_acc:.6f}",
                f"{test_loss:.6f}", f"{test_acc:.6f}", f"{now:.3f}"
            ])
        print(f"[LOG] total_agg={self.t_round} "
              f"avg_train_loss={avg_train_loss:.4f} avg_train_acc={avg_train_acc:.4f} "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} time={now:.1f}s")

        self._log_count += 1
        if self._log_count % 100 == 0:
            path = self.ckpt_dir / f"global_log{self._log_count}_t{self.t_round}.pt"
            torch.save(self.model.state_dict(), path)

        if test_acc >= self.target_accuracy:
            self._stop = True

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
                    self._periodic_eval_and_log()
                next_ts += self.eval_interval_s
        threading.Thread(target=_loop, daemon=True).start()

    def wait(self):
        try:
            while not self.should_stop():
                time.sleep(0.2)
        finally:
            self.mark_stop()

