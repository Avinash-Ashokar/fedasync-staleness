#!/usr/bin/env python3
"""
Hyperparameter tuning script for SqueezeNet on CIFAR-10.
Tests different combinations of learning rate, batch size, and local epochs.
"""

import os
import sys
import yaml
import subprocess
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.helper import set_seed
from utils.model import build_squeezenet
from utils.partitioning import DataDistributor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def build_transform(train: bool = True):
    """Build data transforms for CIFAR-10."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ])


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), 100.0 * correct / total


def run_experiment(
    lr: float,
    batch_size: int,
    local_epochs: int,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    optimizer_type: str = "adam",
    seed: int = 42,
    epochs: int = 50,
    output_dir: Path = None,
) -> Dict:
    """Run a single hyperparameter experiment."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() 
                         else "cpu")
    
    # Load data
    train_transform = build_transform(train=True)
    test_transform = build_transform(train=False)
    
    train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    # Build model
    model = build_squeezenet(num_classes=10, pretrained=False)
    model = model.to(device)
    
    # Setup optimizer
    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_test_acc = 0.0
    best_epoch = 0
    results = []
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
        
        results.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        })
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    return {
        "lr": lr,
        "batch_size": batch_size,
        "local_epochs": local_epochs,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "optimizer": optimizer_type,
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch + 1,
        "final_test_acc": results[-1]["test_acc"],
        "results": results,
    }


def main():
    """Run hyperparameter grid search."""
    print("="*80)
    print("SQUEEZENET HYPERPARAMETER TUNING")
    print("="*80)
    
    # Hyperparameter grid
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    batch_sizes = [32, 64, 128]
    local_epochs_list = [1, 3, 5]
    weight_decays = [0.0, 1e-4, 5e-4]
    optimizers = ["adam", "sgd"]
    
    # Create output directory
    output_dir = Path("hyperparameter_tuning_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{timestamp}.csv"
    summary_file = output_dir / f"summary_{timestamp}.json"
    
    all_results = []
    best_result = None
    best_acc = 0.0
    
    total_experiments = len(learning_rates) * len(batch_sizes) * len(local_epochs_list) * len(weight_decays) * len(optimizers)
    current = 0
    
    print(f"\nTotal experiments: {total_experiments}")
    print(f"Results will be saved to: {results_file}\n")
    
    # Write CSV header
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lr", "batch_size", "local_epochs", "weight_decay", "momentum", "optimizer",
            "best_test_acc", "best_epoch", "final_test_acc"
        ])
    
    # Grid search
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for local_epochs in local_epochs_list:
                for wd in weight_decays:
                    for opt in optimizers:
                        current += 1
                        momentum = 0.9 if opt == "sgd" else 0.0
                        
                        print(f"\n[{current}/{total_experiments}] Testing: lr={lr}, batch={batch_size}, "
                              f"epochs={local_epochs}, wd={wd}, opt={opt}")
                        print("-" * 80)
                        
                        try:
                            result = run_experiment(
                                lr=lr,
                                batch_size=batch_size,
                                local_epochs=local_epochs,
                                weight_decay=wd,
                                momentum=momentum,
                                optimizer_type=opt,
                                epochs=50,
                            )
                            
                            all_results.append(result)
                            
                            # Write to CSV
                            with open(results_file, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    result["lr"],
                                    result["batch_size"],
                                    result["local_epochs"],
                                    result["weight_decay"],
                                    result["momentum"],
                                    result["optimizer"],
                                    f"{result['best_test_acc']:.4f}",
                                    result["best_epoch"],
                                    f"{result['final_test_acc']:.4f}",
                                ])
                            
                            # Track best
                            if result["best_test_acc"] > best_acc:
                                best_acc = result["best_test_acc"]
                                best_result = result
                                print(f"⭐ NEW BEST: {best_acc:.2f}%")
                            
                        except Exception as e:
                            print(f"❌ Error: {e}")
                            continue
    
    # Save summary
    summary = {
        "total_experiments": total_experiments,
        "completed": len(all_results),
        "best_result": best_result,
        "all_results": all_results,
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("="*80)
    print(f"\nTotal experiments: {len(all_results)}")
    print(f"\n🏆 BEST CONFIGURATION:")
    if best_result:
        print(f"   Learning Rate: {best_result['lr']}")
        print(f"   Batch Size: {best_result['batch_size']}")
        print(f"   Local Epochs: {best_result['local_epochs']}")
        print(f"   Weight Decay: {best_result['weight_decay']}")
        print(f"   Optimizer: {best_result['optimizer']}")
        print(f"   Momentum: {best_result['momentum']}")
        print(f"   Best Test Accuracy: {best_result['best_test_acc']:.2f}%")
        print(f"   Best Epoch: {best_result['best_epoch']}")
        print(f"   Final Test Accuracy: {best_result['final_test_acc']:.2f}%")
    
    print(f"\n📊 Results saved to:")
    print(f"   - {results_file}")
    print(f"   - {summary_file}")


if __name__ == "__main__":
    main()

