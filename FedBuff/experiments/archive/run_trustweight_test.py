#!/usr/bin/env python3
"""
Quick test runner for TrustWeight with alpha=1000, 20 rounds.
Thorough logging like FedBuff/FedAsync.
"""

import sys
import os

# Add current directory to path to import from solution.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Extract necessary components from solution.py
# We'll import what we need
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

# Import utilities from solution.py by executing relevant parts
# Actually, let's just create a minimal standalone version

def set_seed(seed: int = 42) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Return the first available computation device (CUDA/MPS/CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Create ResNet-18 adapted for CIFAR-10."""
    if pretrained:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        m = models.resnet18(weights=None)
    # CIFAR-10: 32x32 -> use 3x3 conv, stride 1, no maxpool
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    # Replace classifier
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.num_classes = num_classes
    return m

# Import the actual TrustWeight implementation
# We'll need to import from TrustWeight folder or solution.py
# Let me check what's available
import importlib.util

# Try to load from TrustWeight folder first
spec = importlib.util.spec_from_file_location("trustweight_run", "TrustWeight/run.py")
if spec and spec.loader:
    trustweight_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trustweight_module)
    # Use the run function if available
    print("✅ Loaded TrustWeight from TrustWeight/run.py")
else:
    print("⚠️  Could not load TrustWeight/run.py, will use solution.py approach")
    trustweight_module = None

# Actually, let's use the solution.py approach - convert it to a proper script
# Or better: use the TrustWeight folder directly
print("Using TrustWeight implementation from TrustWeight/ folder...")

# Import from TrustWeight
from TrustWeight.run import main as trustweight_main
from TrustWeight.config import load_config
from TrustWeight.experiment import run_experiment

# Actually, let's create a simple config and run it
def create_test_config():
    """Create test config for alpha=1000, 20 rounds."""
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
        "server": {
            "buffer_size": 5,
            "buffer_timeout_s": 0.0,
            "eta": 0.5,
            "theta": [1.0, -0.1, 0.2],
        },
        "eval": {
            "interval_seconds": 1.0,
            "target_accuracy": 0.8
        },
        "train": {
            "max_rounds": 20
        },
        "partition": {
            "alpha": 1000.0,
            "seed": 1
        },
        "seed": 1,
    }
    return config

if __name__ == "__main__":
    print("="*70)
    print("TrustWeight Test Run: alpha=1000, 20 rounds")
    print("="*70)
    
    # Create test config
    config = create_test_config()
    
    # Create run directory
    run_dir = Path("logs/TrustWeight/test_alpha1000_20rounds") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Run folder: {run_dir}")
    
    # Save config
    with (run_dir / "CONFIG.yaml").open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Write COMMIT.txt
    commit_hash = "test_run"
    csv_header = "total_agg,avg_train_loss,avg_train_acc,test_loss,test_acc,time"
    with (run_dir / "COMMIT.txt").open("w") as f:
        f.write(f"{commit_hash},{csv_header}\n")
    
    print("✅ Configuration saved")
    print(f"\nConfig summary:")
    print(f"  - Alpha: {config['partition']['alpha']}")
    print(f"  - Max rounds: {config['train']['max_rounds']}")
    print(f"  - Clients: {config['clients']['total']}")
    print(f"  - Eta: {config['server']['eta']}")
    print(f"  - Theta: {config['server']['theta']}")
    print(f"\n🚀 Starting experiment...")
    
    # Try to run using TrustWeight/run.py
    try:
        # Update config paths
        config["io"] = {
            "checkpoints_dir": str(run_dir / "checkpoints"),
            "logs_dir": str(run_dir),
            "data_dir": "./data"
        }
        
        # Run experiment
        # This will need to be adapted based on actual TrustWeight API
        print("⚠️  Need to adapt to actual TrustWeight run API")
        print("   For now, please run solution.py notebook with Exp1 modified to 20 rounds")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exception(*sys.exc_info())


