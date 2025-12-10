#!/usr/bin/env python3
"""Quick test script to run TrustWeight for 20 rounds."""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import necessary modules
import time
import random
import threading
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import TrustWeight components
from TrustWeight.client import AsyncClient
from TrustWeight.server import AsyncServer
from TrustWeight.strategy import TrustWeightedAsyncStrategy, TrustWeightedConfig
from utils.partitioning import DataDistributor
from utils.model import build_resnet18
from utils.helper import get_device, set_seed

# Test configuration (based on Exp1 but with 20 rounds)
TEST_CONFIG = {
    "name": "IID (alpha=1000), no stragglers - 20 rounds test",
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
        "eta": 0.5
    },
    "eval": {
        "interval_seconds": 1.0,
        "target_accuracy": 0.8
    },
    "train": {
        "max_rounds": 20  # 20 rounds for testing
    },
    "partition_alpha": 1000.0,  # IID
    "seed": 1,
    "server_runtime": {
        "client_delay": 0.0
    }
}

def print_config(config):
    """Print the configuration being used."""
    print("="*70)
    print("TRUSTWEIGHT TEST CONFIGURATION")
    print("="*70)
    print(f"Experiment: {config['name']}")
    print(f"\nData:")
    print(f"  - Dataset: {config['data']['dataset']}")
    print(f"  - Data dir: {config['data']['data_dir']}")
    print(f"  - Num classes: {config['data']['num_classes']}")
    print(f"  - Partition alpha: {config['partition_alpha']} (IID)")
    print(f"\nClients:")
    print(f"  - Total clients: {config['clients']['total']}")
    print(f"  - Concurrent: {config['clients']['concurrent']}")
    print(f"  - Local epochs: {config['clients']['local_epochs']}")
    print(f"  - Batch size: {config['clients']['batch_size']}")
    print(f"  - Learning rate: {config['clients']['lr']}")
    print(f"  - Momentum: {config['clients']['momentum']}")
    print(f"  - Weight decay: {config['clients']['weight_decay']}")
    print(f"  - Grad clip: {config['clients']['grad_clip']}")
    print(f"  - Stragglers: {config['clients']['struggle_percent']}%")
    print(f"\nTrustWeight Server:")
    print(f"  - Buffer size: {config['trustweight']['buffer_size']}")
    print(f"  - Buffer timeout: {config['trustweight']['buffer_timeout_s']}s")
    print(f"  - Eta (server LR): {config['trustweight']['eta']}")
    print(f"  - Sample weighing: {config['trustweight']['use_sample_weighing']}")
    print(f"\nTraining:")
    print(f"  - Max rounds: {config['train']['max_rounds']}")
    print(f"  - Target accuracy: {config['eval']['target_accuracy']}")
    print(f"  - Eval interval: {config['eval']['interval_seconds']}s")
    print(f"\nOther:")
    print(f"  - Seed: {config['seed']}")
    print("="*70)

def run_test():
    """Run TrustWeight test for 20 rounds."""
    config = TEST_CONFIG
    
    # Print config
    print_config(config)
    
    # Set seed
    seed = config['seed']
    set_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Create output directory
    run_dir = Path("./logs/TrustWeight/test_20rounds") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Run directory: {run_dir}")
    
    # Load and partition data
    print("\n📊 Loading and partitioning data...")
    dd = DataDistributor(
        dataset_name=config['data']['dataset'],
        data_dir=config['data']['data_dir'],
        alpha=config['partition_alpha'],
        num_clients=config['clients']['total'],
        seed=seed
    )
    print(f"✅ Data loaded: {len(dd.train_dataset)} train samples")
    
    # Build global model
    print("\n🏗️  Building global model...")
    global_model = build_resnet18(num_classes=config['data']['num_classes'])
    device = get_device()
    print(f"✅ Model built on device: {device}")
    
    # Create server
    print("\n🖥️  Creating server...")
    server = AsyncServer(
        global_model=global_model,
        total_train_samples=len(dd.train_dataset),
        buffer_size=int(config['trustweight']['buffer_size']),
        buffer_timeout_s=float(config['trustweight']['buffer_timeout_s']),
        use_sample_weighing=bool(config['trustweight']['use_sample_weighing']),
        target_accuracy=float(config['eval']['target_accuracy']),
        max_rounds=int(config['train']['max_rounds']),
        eval_interval_s=int(config['eval']['interval_seconds']),
        data_dir=config['data']['data_dir'],
        checkpoints_dir=str(run_dir / "checkpoints"),
        logs_dir=str(run_dir),
        global_log_csv=str(run_dir / "TrustWeight.csv"),
        client_participation_csv=str(run_dir / "TrustWeightClientParticipation.csv"),
        final_model_path=str(run_dir / "TrustWeightModel.pt"),
        resume=False,
        device=device,
        eta=float(config['trustweight']['eta']),
    )
    print(f"✅ Server created")
    print(f"   - Buffer size: {server.buffer_size}")
    print(f"   - Buffer timeout: {server.buffer_timeout_s}s")
    print(f"   - Eta: {config['trustweight']['eta']}")
    print(f"   - Max rounds: {server.max_rounds}")
    
    # Create clients
    print("\n👥 Creating clients...")
    clients = []
    for cid in range(config['clients']['total']):
        indices = dd.partitions[cid] if cid in dd.partitions else []
        # Convert dict config to GlobalConfig-like object for client
        from types import SimpleNamespace
        cfg = SimpleNamespace()
        cfg.data = SimpleNamespace(
            data_dir=config['data']['data_dir'],
            num_classes=config['data']['num_classes']
        )
        cfg.clients = SimpleNamespace(
            batch_size=config['clients']['batch_size'],
            local_epochs=config['clients']['local_epochs'],
            lr=config['clients']['lr'],
            weight_decay=config['clients']['weight_decay'],
            grad_clip=config['clients']['grad_clip'],
            total=config['clients']['total'],
            struggle_percent=config['clients']['struggle_percent'],
            delay_slow_range=config['clients']['delay_slow_range'],
            delay_fast_range=config['clients']['delay_fast_range'],
            jitter_per_round=config['clients']['jitter_per_round'],
        )
        cfg.server_runtime = SimpleNamespace(
            client_delay=config['server_runtime']['client_delay']
        )
        clients.append(AsyncClient(cid=cid, indices=indices, cfg=cfg))
    print(f"✅ Created {len(clients)} clients")
    
    # Start experiment
    print("\n🚀 Starting experiment...")
    server.start_eval_timer()
    sem = threading.Semaphore(int(config['clients']['concurrent']))
    
    def client_loop(client: AsyncClient):
        while True:
            with sem:
                cont = client.run_once(server)
            if not cont:
                break
            time.sleep(0.05)
    
    # Launch client threads
    threads = []
    for cl in clients:
        t = threading.Thread(target=client_loop, args=(cl,), daemon=False)
        t.start()
        threads.append(t)
    
    print(f"✅ Started {len(threads)} client threads")
    print(f"\n⏳ Running for {config['train']['max_rounds']} rounds...")
    
    # Wait for completion
    start_time = time.time()
    server.wait()
    duration = time.time() - start_time
    
    # Join all threads
    for t in threads:
        t.join(timeout=1.0)
    
    print(f"\n✅ Experiment completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    # Read and display results
    csv_path = run_dir / "TrustWeight.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"\n📊 Results Summary:")
        print(f"   - Total aggregations: {len(df)}")
        print(f"   - Final test accuracy: {df['test_acc'].iloc[-1]:.4f}")
        print(f"   - Best test accuracy: {df['test_acc'].max():.4f}")
        print(f"   - Final train accuracy: {df['avg_train_acc'].iloc[-1]:.4f}")
        print(f"\n📈 First 5 rows:")
        print(df.head().to_string())
        print(f"\n📈 Last 5 rows:")
        print(df.tail().to_string())
    else:
        print("⚠️  No results CSV found")
    
    print(f"\n📁 Results saved to: {run_dir}")
    return run_dir

if __name__ == "__main__":
    try:
        run_dir = run_test()
        print(f"\n✅ Test completed successfully!")
        print(f"   Results: {run_dir}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


