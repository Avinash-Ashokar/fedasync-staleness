# Async FedAsync server with periodic evaluation/logging and accuracy-based stopping
import csv
import time
import threading
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, Future

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from contextlib import redirect_stdout, redirect_stderr
import io

from utils.model import state_to_list, list_to_state, build_resnet18
from utils.helper import get_device


def _testloader(root: str, batch_size: int = 256):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    buf = io.StringIO()
    # Silence torchvision download/cache prints
    with redirect_stdout(buf), redirect_stderr(buf):
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


def _async_eval_worker(
    state_dict: dict,
    data_dir: str,
    num_classes: int,
    log_path: str,
    agg_id: int,
    avg_train_loss: float,
    avg_train_acc: float,
    device_str: str,
) -> Tuple[float, float]:
    """Evaluate a model copy in a background thread to avoid blocking the server."""
    device = torch.device(device_str)
    model = build_resnet18(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model = model.to(device)
    loader = _testloader(root=data_dir, batch_size=256)
    test_loss, test_acc = _evaluate(model, loader, device)

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["total_agg", "avg_train_loss", "avg_train_acc", "test_loss", "test_acc", "time"]
            )
        writer.writerow(
            [
                int(agg_id),
                float(avg_train_loss),
                float(avg_train_acc),
                float(test_loss),
                float(test_acc),
                time.time(),
            ]
        )
    return test_loss, test_acc


class AsyncFedServer:
    """FedAsync: fixed alpha mix (1 - alpha) * w_global + alpha * w_client. Logs every `eval_every_aggs` aggregations."""
    def __init__(
        self,
        global_model: torch.nn.Module,
        alpha: float = 0.5,
        target_accuracy: float = 0.70,
        max_rounds: Optional[int] = None,
        eval_every_aggs: int = 5,
        data_dir: str = "./data",
        logs_dir: str = "./logs",
        global_log_csv: Optional[str] = None,
        client_participation_csv: Optional[str] = None,
        final_model_path: Optional[str] = None,
        num_classes: int = 10,
        device: Optional[torch.device] = None,
    ):
        self.model = global_model
        self.template = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.device = device or get_device()
        self.model.to(self.device)
        self.num_classes = int(num_classes)

        self.alpha = float(alpha)

        self.eval_every_aggs = int(eval_every_aggs)
        self.target_accuracy = float(target_accuracy)
        self.max_rounds = int(max_rounds) if max_rounds is not None else None

        # I/O
        self.data_dir = data_dir
        self.log_dir = Path(logs_dir); self.log_dir.mkdir(parents=True, exist_ok=True)

        # Paths supplied by config (with defaults)
        self.csv_path = Path(global_log_csv) if global_log_csv else (self.log_dir / "FedAsync.csv")
        self.participation_csv = Path(client_participation_csv) if client_participation_csv else (self.log_dir / "FedAsyncClientParticipation.csv")
        self.final_model_path = Path(final_model_path) if final_model_path else Path("./results/FedAsyncModel.pt")
        self.final_model_path.parent.mkdir(parents=True, exist_ok=True)

        # Init CSV headers if files don't exist
        if not self.csv_path.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(
                    ["total_agg", "avg_train_loss", "avg_train_acc", "test_loss", "test_acc", "time"]
                )

        if not self.participation_csv.exists():
            self.participation_csv.parent.mkdir(parents=True, exist_ok=True)
            with self.participation_csv.open("w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "client_id",
                        "local_train_loss",
                        "local_train_acc",
                        "local_test_loss",
                        "local_test_acc",
                        "total_agg",
                        "staleness",
                    ]
                )

        self._lock = threading.Lock()
        self._stop = False
        self.t_round = 0  # increments on every merge
        self.testloader = _testloader(self.data_dir)
        self._train_loss_acc_accum: List[Tuple[float, float, int]] = []  # (loss, acc, n) since last eval
        # Async evaluation state
        self._eval_executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1)
        self._async_eval_future: Optional[Future] = None
        self._init_async_eval_log()
        self._last_eval: Tuple[float, float] = (0.0, 0.0)

    def _save_final_model(self) -> None:
        torch.save(self.model.state_dict(), self.final_model_path)

    def _shutdown_eval_executor(self) -> None:
        """Release async eval thread quickly when stopping."""
        if self._eval_executor is None:
            return
        try:
            self._eval_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self._eval_executor = None

    # ---------- client/server API ----------
    def get_global(self):
        with self._lock:
            return state_to_list(self.model.state_dict()), self.t_round

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
        cleanup_requested = False
        with self._lock:
            if self._stop:
                return
            if self.max_rounds is not None and self.t_round >= self.max_rounds:
                self._stop = True
                cleanup_requested = True
            else:
                # FedAsync merge (fixed alpha, still logs staleness for visibility)
                staleness = max(0, self.t_round - base_version)
                eff = self.alpha

                g = state_to_list(self.model.state_dict())
                merged = [(1.0 - eff) * gi + eff * ci for gi, ci in zip(g, new_params)]
                new_state = list_to_state(self.template, merged)
                self.model.load_state_dict(new_state, strict=True)

                self.t_round += 1

                # Client participation CSV (append staleness like TrustWeight)
                with self.participation_csv.open("a", newline="") as f:
                    csv.writer(f).writerow(
                        [
                            client_id,
                            f"{train_loss:.6f}",
                            f"{train_acc:.6f}",
                            f"{test_loss:.6f}",
                            f"{test_acc:.6f}",
                            self.t_round,
                            float(staleness),
                        ]
                    )

                # accumulate metrics since last eval tick (used by optional timer)
                self._train_loss_acc_accum.append((float(train_loss), float(train_acc), int(num_samples)))
                # Kick off async global eval every eval_every_aggs
                if self.t_round % self.eval_every_aggs == 0:
                    avg_loss, avg_acc = self._compute_avg_train()
                    self._train_loss_acc_accum.clear()
                    self._launch_async_eval_if_needed(self.t_round, avg_loss, avg_acc)

                # only print aggregation number to console
                print(self.t_round)
        if cleanup_requested:
            self._save_final_model()
            self._shutdown_eval_executor()

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop

    def mark_stop(self) -> None:
        with self._lock:
            self._stop = True
            # store final model when marking stop
            self._save_final_model()
        self._shutdown_eval_executor()

    # ---------- evaluation / logging ----------
    def _compute_avg_train(self) -> Tuple[float, float]:
        if not self._train_loss_acc_accum:
            return 0.0, 0.0
        loss_sum, acc_sum, n_sum = 0.0, 0.0, 0
        for l, a, n in self._train_loss_acc_accum:
            loss_sum += l * n
            acc_sum += a * n
            n_sum += n
        return loss_sum / max(1, n_sum), acc_sum / max(1, n_sum)

    def _init_async_eval_log(self) -> None:
        """Ensure the async evaluation CSV exists with header."""
        if self.csv_path.exists():
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["total_agg", "avg_train_loss", "avg_train_acc", "test_loss", "test_acc", "time"]
            )

    def _launch_async_eval_if_needed(self, total_agg: int, avg_train_loss: float, avg_train_acc: float) -> None:
        """Schedule a non-blocking global eval every eval_every_aggs aggregations."""
        if self._eval_executor is None:
            return
        if total_agg % self.eval_every_aggs != 0:
            return
        if self._async_eval_future is not None and not self._async_eval_future.done():
            return

        # snapshot state on current device
        state_copy = {k: v.clone() for k, v in self.model.state_dict().items()}
        self._async_eval_future = self._eval_executor.submit(
            _async_eval_worker,
            state_copy,
            self.data_dir,
            self.num_classes,
            str(self.csv_path),
            total_agg,
            avg_train_loss,
            avg_train_acc,
            str(self.device),
        )
        self._async_eval_future.add_done_callback(self._handle_eval_result)

    def _handle_eval_result(self, fut: Future) -> None:
        try:
            test_loss, test_acc = fut.result()
        except Exception:
            return
        with self._lock:
            self._last_eval = (float(test_loss), float(test_acc))
            if test_acc >= self.target_accuracy:
                self._stop = True
                self._save_final_model()
                self._shutdown_eval_executor()

    def wait(self):
        try:
            while not self.should_stop():
                time.sleep(0.2)
        finally:
            self.mark_stop()
