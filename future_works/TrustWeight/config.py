from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


# ----------------------------- dataclasses -----------------------------------


@dataclass
class DataConfig:
    dataset: str
    data_dir: str
    num_classes: int


@dataclass
class ClientsConfig:
    total: int
    concurrent: int
    local_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    struggle_percent: int
    delay_slow_range: Tuple[float, float]
    delay_fast_range: Tuple[float, float]
    jitter_per_round: float
    fix_delays_per_client: bool


@dataclass
class EvalConfig:
    interval_seconds: int
    target_accuracy: float


@dataclass
class TrainConfig:
    max_rounds: int
    update_clip_norm: float


@dataclass
class ServerRuntimeConfig:
    client_delay: float


@dataclass
class IOConfig:
    checkpoints_dir: str
    logs_dir: str
    results_dir: str
    global_log_csv: str
    client_participation_csv: str
    final_model_path: str


@dataclass
class GlobalConfig:
    data: DataConfig
    clients: ClientsConfig
    eval: EvalConfig
    train: TrainConfig
    partition_alpha: float
    seed: int
    server_runtime: ServerRuntimeConfig
    io: IOConfig


# ----------------------------- loader ----------------------------------------


def _as_tuple(x: Any) -> Tuple[float, float]:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return float(x[0]), float(x[1])
    raise ValueError(f"Expected length-2 sequence, got {x!r}")


def load_config(path: str | Path = "TrustWeight/config.yaml") -> GlobalConfig:
    path = Path(path)
    with path.open("r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    data_cfg = DataConfig(
        dataset=raw["data"]["dataset"],
        data_dir=raw["data"]["data_dir"],
        num_classes=int(raw["data"]["num_classes"]),
    )

    clients_section = raw["clients"]
    clients_cfg = ClientsConfig(
        total=int(clients_section["total"]),
        concurrent=int(clients_section["concurrent"]),
        local_epochs=int(clients_section["local_epochs"]),
        batch_size=int(clients_section["batch_size"]),
        lr=float(clients_section["lr"]),
        weight_decay=float(clients_section.get("weight_decay", 5e-4)),
        grad_clip=float(clients_section.get("grad_clip", 5.0)),
        struggle_percent=int(clients_section.get("struggle_percent", 0)),
        delay_slow_range=_as_tuple(clients_section.get("delay_slow_range", [0.8, 2.0])),
        delay_fast_range=_as_tuple(clients_section.get("delay_fast_range", [0.0, 0.2])),
        jitter_per_round=float(clients_section.get("jitter_per_round", 0.0)),
        fix_delays_per_client=bool(clients_section.get("fix_delays_per_client", True)),
    )

    eval_cfg = EvalConfig(
        interval_seconds=int(raw["eval"]["interval_seconds"]),
        target_accuracy=float(raw["eval"]["target_accuracy"]),
    )

    train_cfg = TrainConfig(
        max_rounds=int(raw["train"]["max_rounds"]),
        update_clip_norm=float(raw["train"].get("update_clip_norm", 10.0)),
    )

    server_runtime_cfg = ServerRuntimeConfig(
        client_delay=float(raw["server_runtime"].get("client_delay", 0.0))
    )

    io_section = raw["io"]
    io_cfg = IOConfig(
        checkpoints_dir=io_section["checkpoints_dir"],
        logs_dir=io_section["logs"],
        results_dir=io_section["results"],
        global_log_csv=io_section["global_log_csv"],
        client_participation_csv=io_section["client_participation_csv"],
        final_model_path=io_section["final_model_path"],
    )

    cfg = GlobalConfig(
        data=data_cfg,
        clients=clients_cfg,
        eval=eval_cfg,
        train=train_cfg,
        partition_alpha=float(raw.get("partition_alpha", 0.5)),
        seed=int(raw.get("seed", 42)),
        server_runtime=server_runtime_cfg,
        io=io_cfg,
    )
    return cfg
