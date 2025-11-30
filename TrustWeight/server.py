# Asynchronous federated server implementing the Trust-Weighted projection rule
import csv
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, OrderedDict as ODType

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.model import build_resnet18
from utils.helper import get_device

from .config import GlobalConfig, load_config
from .strategy import TrustWeightedAsyncStrategy


# ---------------------------------------------------------------------------

def _testloader(root: str, batch_size: int = 256) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    ds = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    # num_workers=0 to avoid multiprocessing / SemLock issues
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _flatten_state(state: "ODType[str, torch.Tensor]") -> torch.Tensor:
    """Flatten a state_dict into a 1D tensor.

    The default implementation simply concatenates parameters in whatever
    iteration order the dictionary exposes.  While this works for
    ``OrderedDict`` instances created from the same model template, it can
    become brittle when ``state`` is a plain Python ``dict`` because the
    insertion order may differ from the server's canonical template.  An
    inconsistent ordering corrupts the mapping between parameter slices and
    model layers, which in turn leads to meaningless update vectors and
    prevents the global model from learning.

    To avoid such subtle bugs, prefer ``_flatten_state_by_template`` when
    flattening a state that may not share the exact ordering with the
    server's template (see ``_flatten_state_by_template`` below).
    """
    return torch.cat([p.reshape(-1) for p in state.values()])


def _flatten_state_by_template(
    state: Dict[str, torch.Tensor], template: "ODType[str, torch.Tensor]"
) -> torch.Tensor:
    """Flatten ``state`` according to the key ordering of ``template``.

    Many downstream computations (e.g. computing parameter deltas) assume
    that all 1D parameter vectors follow the same ordering.  When
    ``state`` is a plain ``dict`` created from a model state_dict,
    Python preserves insertion order, but nothing guarantees that this
    ordering matches that of the server's ``template``.  This helper
    constructs a flattened tensor by explicitly iterating over the keys of
    ``template``, thereby aligning the parameter ordering regardless of
    how ``state`` was constructed.

    Args:
        state: Mapping from parameter names to tensors.  Can be a
            standard ``dict`` or ``OrderedDict``.
        template: The canonical parameter ordering to follow.

    Returns:
        A 1D tensor containing all parameters from ``state`` in the order
        specified by ``template``.
    """
    return torch.cat([state[k].reshape(-1) for k in template.keys()])


def _vector_to_state(
    vec: torch.Tensor, template: "ODType[str, torch.Tensor]"
) -> "ODType[str, torch.Tensor]":
    new_state: "ODType[str, torch.Tensor]" = type(template)()
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
    new_params: "ODType[str, torch.Tensor]"
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
    """Central server maintaining global model and asynchronous buffer."""

    def __init__(self, cfg: Optional[GlobalConfig] = None) -> None:
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.device = get_device()

        # data / evaluation
        self.testloader = _testloader(cfg.data.data_dir, batch_size=256)

        # global model and version history
        model = build_resnet18(num_classes=cfg.data.num_classes)
        self._template_state: ODType[str, torch.Tensor] = ODType(
            (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
        )
        self._global_state: ODType[str, torch.Tensor] = ODType(
            (k, v.clone()) for k, v in self._template_state.items()
        )

        self._model_versions: List[ODType[str, torch.Tensor]] = [
            ODType((k, v.clone()) for k, v in self._global_state.items())
        ]
        self._version: int = 0

        # strategy encapsulating all math from the PDF
        dim = _flatten_state(self._global_state).numel()
        self.strategy = TrustWeightedAsyncStrategy(dim=dim)

        # async buffer
        self.buffer: List[ClientUpdateState] = []
        self.buffer_size: int = 5  # K
        self.buffer_timeout_s: float = 5.0  # Δt in seconds
        self._last_flush_ts: float = time.time()

        # logging / control
        self.io_root = Path(cfg.io.logs_dir)
        self.io_root.mkdir(parents=True, exist_ok=True)

        # Global training log CSV
        self.global_log_path = Path(cfg.io.global_log_csv)
        self.global_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_global_log()

        # Client participation log CSV
        self.client_log_path = Path(cfg.io.client_participation_csv)
        self.client_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_client_log()

        self.eval_interval_s: float = cfg.eval.interval_seconds
        self.target_accuracy: float = cfg.eval.target_accuracy
        self.max_rounds: int = cfg.train.max_rounds
        self.update_clip_norm: float = float(cfg.train.update_clip_norm)

        self._num_aggregations: int = 0  # total_agg counter
        self._stop: bool = False
        self._stop_reason: str = ""
        self._lock = threading.Lock()

        # Ensures only one aggregation is in-flight at a time.  Without
        # this lock, multiple threads could concurrently call
        # ``_aggregate`` on separate buffers, causing race conditions in
        # versioning and inconsistent staleness calculations.  All calls
        # to ``_aggregate`` must acquire this lock.
        self._agg_lock = threading.Lock()

    # ----------------------------------------------------------------- logging

    def _init_global_log(self) -> None:
        """Initialize the global training CSV if it does not exist.

        Columns (per aggregation):
            total_agg, avg_train_loss, avg_train_acc, test_loss, test_acc, time
        """
        if self.global_log_path.exists():
            return
        with self.global_log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "total_agg",
                    "avg_train_loss",
                    "avg_train_acc",
                    "test_loss",
                    "test_acc",
                    "time",
                ]
            )

    def _append_global_log(
        self,
        total_agg: int,
        avg_train_loss: float,
        avg_train_acc: float,
        test_loss: float,
        test_acc: float,
    ) -> None:
        """Append a single aggregation row to the global CSV."""
        ts = time.time()
        with self.global_log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    int(total_agg),
                    float(avg_train_loss),
                    float(avg_train_acc),
                    float(test_loss),
                    float(test_acc),
                    ts,
                ]
            )

    def _init_client_log(self) -> None:
        """Initialize the client participation CSV.

        Columns:
            client_id, local_train_loss, local_train_acc,
            local_test_loss, local_test_acc, total_agg, staleness
        """
        if self.client_log_path.exists():
            return
        with self.client_log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
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

    def _append_client_participation_log(
        self,
        total_agg: int,
        updates: List[ClientUpdateState],
        staleness_list: List[float],
    ) -> None:
        """Append one row per client update participating in this aggregation.

        staleness_list[i] is the τ_i for updates[i].
        """
        with self.client_log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            for u, tau_i in zip(updates, staleness_list):
                writer.writerow(
                    [
                        int(u.client_id),
                        float(u.loss_after),
                        float(u.train_acc),
                        float(u.test_loss),
                        float(u.test_acc),
                        int(total_agg),
                        float(tau_i),
                    ]
                )

    # ------------------------------------------------------------------- public

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

    # --------------------------------------------------------------- model I/O

    def get_global_model(self) -> Tuple[int, Dict[str, torch.Tensor]]:
        """Return (version, state_dict) of the current global model."""
        with self._lock:
            version = self._version
            # Preserve the parameter ordering when sending to clients by
            # constructing an OrderedDict.  A plain dict may reorder keys
            # unexpectedly on some Python versions or implementations,
            # breaking downstream flatten/unflatten assumptions.  Clones
            # ensure the server's tensors remain unmodified.
            state = ODType((k, v.clone()) for k, v in self._global_state.items())
        return version, state

    # --------------------------------------------------------------- evaluation

    def _make_model_from_state(self, state: Dict[str, torch.Tensor]) -> torch.nn.Module:
        model = build_resnet18(num_classes=self.cfg.data.num_classes)
        model.load_state_dict(state)
        return model.to(self.device)

    def _evaluate_global(self) -> Tuple[float, float]:
        """Evaluate the current global model on the test set."""
        model = self._make_model_from_state(self._global_state)
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

    # --------------------------------------------------------------- aggregation

    def _flush_buffer_if_needed(self) -> None:
        now = time.time()
        should_flush = False
        if len(self.buffer) >= self.buffer_size:
            should_flush = True
        elif (now - self._last_flush_ts) >= self.buffer_timeout_s and self.buffer:
            should_flush = True

        if not should_flush:
            return

        # copy buffer locally under lock then release for heavy work
        with self._lock:
            buffer_copy = list(self.buffer)
            self.buffer.clear()
            self._last_flush_ts = now

        # Serialize aggregations to avoid version races
        with self._agg_lock:
            self._aggregate(buffer_copy)

    def _aggregate(self, updates: List[ClientUpdateState]) -> None:
        """Aggregate a batch of client updates and log to CSVs."""
        if not updates:
            return

        # Snapshot of current global parameters and version history
        with self._lock:
            # Always flatten the global state according to the template order
            global_vec = _flatten_state_by_template(self._global_state, self._template_state)
            version_now = self._version
            model_versions = list(self._model_versions)

        # Construct per-update vectors and metadata for the strategy,
        # and collect staleness τ_i for logging.
        update_vectors: List[Dict[str, torch.Tensor]] = []
        staleness_list: List[float] = []
        valid_updates: List[ClientUpdateState] = []

        for u in updates:
            base_state = model_versions[u.base_version]
            # Flatten base_state and new_params using the canonical template order
            base_vec = _flatten_state_by_template(base_state, self._template_state)
            new_vec = _flatten_state_by_template(u.new_params, self._template_state)
            ui = new_vec - base_vec

            # Skip obviously bad updates (NaN/Inf) to avoid corrupting the global model
            if not torch.isfinite(ui).all():
                print(f"[Server] Dropping client {u.client_id} update due to NaN/Inf values")
                continue
            if self.update_clip_norm > 0:
                norm = torch.norm(ui)
                if torch.isfinite(norm) and norm.item() > self.update_clip_norm:
                    ui = ui * (self.update_clip_norm / (norm + 1e-12))

            # τ_i = current-server-version - base_version (same τ_i used in strategy)
            tau_i = float(max(0, version_now - u.base_version))
            staleness_list.append(tau_i)

            delta_loss = float(u.delta_loss)  # ΔL̃_i
            update_vectors.append(
                {
                    "u": ui,
                    "tau": torch.tensor(tau_i, dtype=torch.float32),
                    "num_samples": torch.tensor(float(u.num_samples), dtype=torch.float32),
                    "delta_loss": torch.tensor(delta_loss, dtype=torch.float32),
                }
            )
            valid_updates.append(u)

        if not update_vectors:
            print("[Server] Buffer flush skipped: no valid updates after filtering.")
            return

        # Run the trust-weighted aggregation strategy (unchanged algorithm)
        new_global_vec, agg_metrics = self.strategy.aggregate(global_vec, update_vectors)

        # Map back into parameter state_dict form
        new_state = _vector_to_state(new_global_vec, self._template_state)

        # Compute average local train metrics for this aggregation
        avg_train_loss = sum(u.loss_after for u in valid_updates) / len(valid_updates)
        avg_train_acc = sum(u.train_acc for u in valid_updates) / len(valid_updates)

        # Commit the new global model and update aggregation counter
        with self._lock:
            self._global_state = ODType((k, v.clone()) for k, v in new_state.items())
            self._model_versions.append(
                ODType((k, v.clone()) for k, v in self._global_state.items())
            )
            self._version = len(self._model_versions) - 1
            self._num_aggregations += 1
            total_agg = self._num_aggregations

        # Evaluate updated global model on test data
        test_loss, test_acc = self._evaluate_global()

        # Log global metrics and per-client participation (now with staleness)
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

        # Stopping conditions based on global performance and max rounds
        if test_acc >= self.target_accuracy:
            self.mark_stop(f"target accuracy {test_acc:.4f} reached")
        if total_agg >= self.max_rounds:
            self.mark_stop("max aggregation rounds reached")

    # ------------------------------------------------------------ client entry

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
        """Entry point called by clients after local training.

        This method only enqueues the update and triggers buffer flushing.
        All CSV logging tied to a particular aggregation happens inside
        `_aggregate` so that `total_agg` and `staleness` are consistent.
        """
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

    # --------------------------------------------------------------- lifecycle

    def wait(self) -> None:
        """Block until training is finished (stopping condition reached)."""
        try:
            while not self.should_stop():
                time.sleep(0.2)
        finally:
            self.mark_stop(self._stop_reason or "training finished")
