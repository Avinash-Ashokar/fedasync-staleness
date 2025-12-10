#!/usr/bin/env python3
"""
Quick test runner for TrustWeight: alpha=1000, 20 rounds with thorough logging.
Extracts and runs the experiment from solution.py.
"""

import sys
import os
import subprocess
import json

# Since solution.py is a Jupyter notebook format, we'll convert it to a script
# or use jupyter nbconvert, or better: extract the key function and run it

print("="*70)
print("TrustWeight Quick Test: alpha=1000, 20 rounds")
print("="*70)
print("\nThis will run Exp1 from solution.py with 20 rounds.")
print("Make sure solution.py has Exp1 configured for 20 rounds.\n")

# Check if solution.py exists
if not os.path.exists("solution.py"):
    print("❌ Error: solution.py not found!")
    sys.exit(1)

# Read solution.py and extract the run_single_experiment function
# Actually, since it's a notebook, let's use a different approach:
# Create a minimal script that imports from TrustWeight directly

print("✅ Creating standalone runner script...")

standalone_script = '''#!/usr/bin/env python3
"""Standalone TrustWeight runner for alpha=1000, 20 rounds."""

import sys
import os
import time
import csv
import threading
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import yaml

# Add TrustWeight to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import TrustWeight components
from TrustWeight.client import AsyncClient
from TrustWeight.server import AsyncServer
from TrustWeight.strategy import TrustWeightedAsyncStrategy, TrustWeightedConfig
from utils.model import build_resnet18
from utils.partitioning import DataDistributor
from utils.helper import set_seed, get_device

def main():
    print("="*70)
    print("TrustWeight Test: alpha=1000, 20 rounds")
    print("="*70)
    
    # Configuration
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
            "delay_slow_range": [0.8, 2.0],
            "delay_fast_range": [0.0, 0.2],
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
    
    print(f"\\n✅ Run folder: {run_dir}")
    
    # Write COMMIT.txt
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()[:8]
    if not commit_hash:
        commit_hash = "test_run"
    csv_header = "total_agg,avg_train_loss,avg_train_acc,test_loss,test_acc,time"
    with (run_dir / "COMMIT.txt").open("w") as f:
        f.write(f"{commit_hash},{csv_header}\\n")
    
    # Save config
    with (run_dir / "CONFIG.yaml").open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("✅ Configuration saved")
    print(f"\\nConfig:")
    print(f"  - Alpha: {config['partition_alpha']}")
    print(f"  - Max rounds: {config['train']['max_rounds']}")
    print(f"  - Clients: {config['clients']['total']}")
    print(f"  - Eta: {config['trustweight']['eta']}")
    print(f"  - Theta: {config['trustweight']['theta']}")
    
    # Load and partition data
    print(f"\\n📊 Loading and partitioning data...")
    dd = DataDistributor(dataset_name=config["data"]["dataset"], data_dir=config["data"]["data_dir"])
    dd.distribute_data(
        num_clients=int(config["clients"]["total"]),
        alpha=float(config["partition_alpha"]),
        seed=seed
    )
    print(f"✅ Data partitioned: {config['clients']['total']} clients, alpha={config['partition_alpha']}")
    
    # Build global model
    print(f"\\n🏗️  Building global model...")
    global_model = build_resnet18(num_classes=config["data"]["num_classes"], pretrained=False)
    print(f"✅ Model created: ResNet-18")
    
    # Initialize server
    print(f"\\n🖥️  Initializing server...")
    server = AsyncServer(
        global_model=global_model,
        total_train_samples=len(dd.train_dataset),
        buffer_size=int(config["trustweight"]["buffer_size"]),
        buffer_timeout_s=float(config["trustweight"]["buffer_timeout_s"]),
        use_sample_weighing=bool(config["trustweight"]["use_sample_weighing"]),
        target_accuracy=float(config["eval"]["target_accuracy"]),
        max_rounds=int(config["train"]["max_rounds"]),
        eval_interval_s=int(config["eval"]["interval_seconds"]),
        data_dir=config["data"]["data_dir"],
        checkpoints_dir=str(run_dir / "checkpoints"),
        logs_dir=str(run_dir),
        global_log_csv=str(run_dir / "TrustWeight.csv"),
        client_participation_csv=str(run_dir / "TrustWeightClientParticipation.csv"),
        final_model_path=str(run_dir / "TrustWeightModel.pt"),
        resume=False,
        device=get_device(),
        eta=float(config["trustweight"].get("eta", 0.5)),
        theta=tuple(config["trustweight"].get("theta", [1.0, -0.1, 0.2])),
    )
    print(f"✅ Server initialized (device: {server.device})")
    
    # Setup clients
    print(f"\\n👥 Setting up clients...")
    n = int(config["clients"]["total"])
    clients: List[AsyncClient] = []
    for cid in range(n):
        indices = dd.partitions[cid] if cid in dd.partitions else []
        clients.append(
            AsyncClient(
                cid=cid,
                indices=indices,
                cfg=config,
            )
        )
    num_slow = sum(1 for c in clients if getattr(c, "is_slow", False))
    num_fast = len(clients) - num_slow
    print(f"✅ Created {len(clients)} clients ({num_slow} slow, {num_fast} fast)")
    
    # Start experiment
    print(f"\\n🚀 Starting experiment...")
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
    print(f"\\n⏳ Running for {config['train']['max_rounds']} rounds...")
    print("   (Progress will be logged to CSV files)")
    
    # Wait for completion
    start_time = time.time()
    server.wait()
    for t in threads:
        t.join()
    
    elapsed = time.time() - start_time
    
    print(f"\\n✅ Experiment completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"\\n📁 Results saved to: {run_dir}")
    print(f"   - TrustWeight.csv: Global metrics")
    print(f"   - TrustWeightClientParticipation.csv: Client participation")
    print(f"   - CONFIG.yaml: Configuration used")
    print(f"   - COMMIT.txt: Run metadata")
    
    # Print summary
    csv_path = run_dir / "TrustWeight.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"\\n📊 Summary:")
        print(f"   - Total rounds: {df['total_agg'].max()}")
        print(f"   - Final test accuracy: {df['test_acc'].iloc[-1]:.4f}")
        print(f"   - Best test accuracy: {df['test_acc'].max():.4f}")
        print(f"   - Total time: {df['time'].iloc[-1]:.2f} seconds")

if __name__ == "__main__":
    main()
'''

with open("run_trustweight_standalone.py", "w") as f:
    f.write(standalone_script)

print("✅ Created run_trustweight_standalone.py")
print("\nNow running the test...\n")

# Run it
os.chmod("run_trustweight_standalone.py", 0o755)
subprocess.run([sys.executable, "run_trustweight_standalone.py"])


