#!/usr/bin/env python3
"""
TrustWeight test run: alpha=1000, 20 rounds with thorough logging.
Uses TrustWeight implementation directly.
"""

import os
import sys
import time
import csv
import threading
import random
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List

# Setup environment
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import torch
import yaml

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from TrustWeight.client import AsyncClient
from TrustWeight.server import AsyncServer
from TrustWeight.config import (
    GlobalConfig, DataConfig, ClientsConfig, EvalConfig, 
    TrainConfig, ServerRuntimeConfig, IOConfig
)
from TrustWeight.strategy import TrustWeightedAsyncStrategy
from utils.model import build_resnet18
from utils.partitioning import DataDistributor
from utils.helper import set_seed, get_device

def main():
    print("="*70)
    print("TrustWeight Test Run: alpha=1000, 20 rounds")
    print("="*70)
    
    # Configuration matching Exp1 from solution.py
    config = {
        "data": {
            "dataset": "cifar10",
            "data_dir": "./data",
            "num_classes": 10
        },
        "clients": {
            "total": 20,
            "concurrent": 5,
            "local_epochs": 1,
            "batch_size": 128,
            "lr": 0.005,
            "momentum": 0.0,
            "weight_decay": 0.001,
            "grad_clip": 5.0,
            "struggle_percent": 0,
            "delay_slow_range": [0.0, 0.0],
            "delay_fast_range": [0.0, 0.0],
            "jitter_per_round": 0.0,
        },
        "trustweight": {
            "buffer_size": 5,
            "buffer_timeout_s": 0.0,
            "use_sample_weighing": True,
            "eta": 0.5,
            "theta": [1.0, -0.1, 0.2]
        },
        "eval": {
            "interval_seconds": 1.0,
            "target_accuracy": 0.8
        },
        "train": {
            "max_rounds": 20
        },
        "partition_alpha": 1000.0,
        "seed": 1,
        "server_runtime": {
            "client_delay": 0.0
        }
    }
    
    # Set seed
    seed = int(config.get("seed", 1))
    set_seed(seed)
    random.seed(seed)
    
    # Create timestamped run folder
    run_dir = Path("logs/TrustWeight/test_alpha1000_20rounds") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✅ Run folder: {run_dir}")
    
    # Write COMMIT.txt
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL
        ).decode().strip()[:8]
    except:
        commit_hash = "test_run"
    
    csv_header = "total_agg,avg_train_loss,avg_train_acc,test_loss,test_acc,time"
    with (run_dir / "COMMIT.txt").open("w") as f:
        f.write(f"{commit_hash},{csv_header}\n")
    
    # Save config
    with (run_dir / "CONFIG.yaml").open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("✅ Configuration saved")
    print(f"\n📋 Configuration:")
    print(f"   - Alpha: {config['partition_alpha']}")
    print(f"   - Max rounds: {config['train']['max_rounds']}")
    print(f"   - Clients: {config['clients']['total']} (concurrent: {config['clients']['concurrent']})")
    print(f"   - Local epochs: {config['clients']['local_epochs']}")
    print(f"   - Learning rate: {config['clients']['lr']}")
    print(f"   - Eta (server): {config['trustweight']['eta']}")
    print(f"   - Theta: {config['trustweight']['theta']}")
    print(f"   - Buffer size: {config['trustweight']['buffer_size']}")
    
    # Load and partition data
    print(f"\n📊 Loading and partitioning data...")
    dd = DataDistributor(
        dataset_name=config["data"]["dataset"], 
        data_dir=config["data"]["data_dir"]
    )
    dd.distribute_data(
        num_clients=int(config["clients"]["total"]),
        alpha=float(config["partition_alpha"]),
        seed=seed
    )
    print(f"✅ Data partitioned: {config['clients']['total']} clients, alpha={config['partition_alpha']}")
    for i in range(min(3, config['clients']['total'])):
        print(f"   Client {i}: {len(dd.partitions[i])} samples")
    
    # Create GlobalConfig object
    print(f"\n🖥️  Creating server configuration...")
    cfg = GlobalConfig(
        data=DataConfig(
            dataset=config["data"]["dataset"],
            data_dir=config["data"]["data_dir"],
            num_classes=config["data"]["num_classes"]
        ),
        clients=ClientsConfig(
            total=config["clients"]["total"],
            concurrent=config["clients"]["concurrent"],
            local_epochs=config["clients"]["local_epochs"],
            batch_size=config["clients"]["batch_size"],
            lr=config["clients"]["lr"],
            weight_decay=config["clients"]["weight_decay"],
            grad_clip=config["clients"]["grad_clip"],
            struggle_percent=config["clients"]["struggle_percent"],
            delay_slow_range=tuple(config["clients"]["delay_slow_range"]),
            delay_fast_range=tuple(config["clients"]["delay_fast_range"]),
            jitter_per_round=config["clients"]["jitter_per_round"],
            fix_delays_per_client=True,
        ),
        eval=EvalConfig(
            interval_seconds=config["eval"]["interval_seconds"],
            target_accuracy=config["eval"]["target_accuracy"]
        ),
        train=TrainConfig(
            max_rounds=config["train"]["max_rounds"],
            update_clip_norm=5.0,
        ),
        partition_alpha=config["partition_alpha"],
        seed=config["seed"],
        server_runtime=ServerRuntimeConfig(
            client_delay=config["server_runtime"]["client_delay"]
        ),
        io=IOConfig(
            checkpoints_dir=str(run_dir / "checkpoints"),
            logs_dir=str(run_dir),
            results_dir=str(run_dir),
            global_log_csv=str(run_dir / "TrustWeight.csv"),
            client_participation_csv=str(run_dir / "TrustWeightClientParticipation.csv"),
            final_model_path=str(run_dir / "TrustWeightModel.pt"),
        )
    )
    
    # Set server config for theta (this needs to be done via strategy config)
    # The TrustWeight server uses TrustWeightedAsyncStrategy which needs theta
    # Let's check if we need to modify the strategy after server creation
    
    # Initialize server
    print(f"✅ Server configuration created")
    server = AsyncServer(cfg=cfg)
    
    # Set theta in strategy (TrustWeight server doesn't expose this in config yet)
    from TrustWeight.strategy import TrustWeightedConfig
    theta_tuple = tuple(config["trustweight"].get("theta", [1.0, -0.1, 0.2]))
    strategy_cfg = TrustWeightedConfig(eta=config["trustweight"]["eta"], theta=theta_tuple)
    dim = server.strategy.dim
    server.strategy = TrustWeightedAsyncStrategy(dim=dim, cfg=strategy_cfg)
    
    print(f"✅ Server initialized (device: {server.device})")
    print(f"   - Eta: {config['trustweight']['eta']}")
    print(f"   - Theta: {theta_tuple}")
    
    # Setup clients
    print(f"\n👥 Setting up clients...")
    n = int(config["clients"]["total"])
    clients: List[AsyncClient] = []
    for cid in range(n):
        indices = dd.partitions[cid] if cid in dd.partitions else []
        clients.append(
            AsyncClient(
                cid=cid,
                indices=indices,
                cfg=cfg,  # Use GlobalConfig object, not dict
            )
        )
    num_slow = sum(1 for c in clients if getattr(c, "is_slow", False))
    num_fast = len(clients) - num_slow
    print(f"✅ Created {len(clients)} clients ({num_slow} slow, {num_fast} fast)")
    
    # Start experiment
    print(f"\n🚀 Starting experiment...")
    server.start_eval_timer()
    sem = threading.Semaphore(int(config["clients"]["concurrent"]))
    
    def client_loop(client: AsyncClient):
        while True:
            with sem:
                cont = client.run_once(server)
            if not cont:
                break
            time.sleep(0.05)
    
    threads = []
    for cl in clients:
        t = threading.Thread(target=client_loop, args=(cl,), daemon=False)
        t.start()
        threads.append(t)
    
    print(f"✅ Started {len(threads)} client threads")
    print(f"\n⏳ Running for {config['train']['max_rounds']} rounds...")
    print("   (Progress will be logged to CSV files in real-time)")
    print(f"   Logs: {run_dir / 'TrustWeight.csv'}")
    
    # Wait for completion
    start_time = time.time()
    server.wait()
    for t in threads:
        t.join()
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Experiment completed!")
    print(f"   Duration: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"\n📁 Results saved to: {run_dir}")
    print(f"   - TrustWeight.csv: Global metrics")
    print(f"   - TrustWeightClientParticipation.csv: Client participation")
    print(f"   - CONFIG.yaml: Configuration used")
    print(f"   - COMMIT.txt: Run metadata")
    
    # Print summary
    csv_path = run_dir / "TrustWeight.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"\n📊 Summary:")
        print(f"   - Total rounds: {int(df['total_agg'].max())}")
        print(f"   - Final test accuracy: {df['test_acc'].iloc[-1]:.4f}")
        print(f"   - Best test accuracy: {df['test_acc'].max():.4f} (round {int(df.loc[df['test_acc'].idxmax(), 'total_agg'])})")
        print(f"   - Final train accuracy: {df['avg_train_acc'].iloc[-1]:.4f}")
        print(f"   - Total time: {df['time'].iloc[-1]:.2f} seconds ({df['time'].iloc[-1]/60:.2f} minutes)")
        print(f"\n   First 3 rows:")
        print(df.head(3).to_string(index=False))
        print(f"\n   Last 3 rows:")
        print(df.tail(3).to_string(index=False))
    else:
        print(f"\n⚠️  CSV file not found at {csv_path}")

if __name__ == "__main__":
    main()

