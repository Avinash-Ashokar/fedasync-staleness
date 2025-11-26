#!/usr/bin/env python3
"""
Quick hyperparameter tuning for SqueezeNet - smaller grid for faster results.
"""

import os
import sys
import csv
import json
from pathlib import Path
from datetime import datetime
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent))
from utils.model import build_squeezenet


def build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    return train_transform, test_transform


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return running_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return running_loss / len(loader), 100.0 * correct / total


def run_experiment(config, device, epochs=30):
    lr = config["lr"]
    batch_size = config["batch_size"]
    weight_decay = config["weight_decay"]
    optimizer_type = config["optimizer"]
    momentum = config.get("momentum", 0.9)
    seed = config.get("seed", 42)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    train_transform, test_transform = build_transforms()
    train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    model = build_squeezenet(num_classes=10, pretrained=False)
    model = model.to(device)
    
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    criterion = nn.CrossEntropyLoss()
    best_test_acc = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: Train={train_acc:5.2f}%, Test={test_acc:5.2f}%")
    
    return {
        **config,
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "final_test_acc": test_acc,
        "final_train_acc": train_acc,
    }


def main():
    print("="*80)
    print("SQUEEZENET QUICK HYPERPARAMETER TUNING")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() 
                         else "cpu")
    print(f"\nDevice: {device}")
    
    # Smaller grid for quick testing
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    batch_sizes = [64, 128]
    weight_decays = [0.0, 5e-4]
    optimizers = ["adam", "sgd"]
    
    configs = []
    for lr, bs, wd, opt in product(learning_rates, batch_sizes, weight_decays, optimizers):
        configs.append({
            "lr": lr,
            "batch_size": bs,
            "weight_decay": wd,
            "optimizer": opt,
            "momentum": 0.9 if opt == "sgd" else 0.0,
            "seed": 42,
        })
    
    total = len(configs)
    print(f"\nTotal experiments: {total} (30 epochs each)")
    
    output_dir = Path("hyperparameter_tuning_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = output_dir / f"quick_results_{timestamp}.csv"
    json_file = output_dir / f"quick_summary_{timestamp}.json"
    
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lr", "batch_size", "weight_decay", "optimizer", "momentum",
            "best_test_acc", "best_epoch", "final_test_acc", "final_train_acc"
        ])
    
    results = []
    best_result = None
    best_acc = 0.0
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{total}] lr={config['lr']}, batch={config['batch_size']}, "
              f"wd={config['weight_decay']}, opt={config['optimizer']}")
        print("-" * 80)
        
        try:
            result = run_experiment(config, device, epochs=30)
            results.append(result)
            
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    result["lr"], result["batch_size"], result["weight_decay"],
                    result["optimizer"], result["momentum"],
                    f"{result['best_test_acc']:.4f}", result["best_epoch"],
                    f"{result['final_test_acc']:.4f}", f"{result['final_train_acc']:.4f}",
                ])
            
            if result["best_test_acc"] > best_acc:
                best_acc = result["best_test_acc"]
                best_result = result
                print(f"⭐ NEW BEST: {best_acc:.2f}%")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    summary = {
        "total_experiments": total,
        "completed": len(results),
        "best_result": best_result,
        "all_results": sorted(results, key=lambda x: x["best_test_acc"], reverse=True),
    }
    
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("QUICK TUNING COMPLETE")
    print("="*80)
    print(f"\nCompleted: {len(results)}/{total}")
    
    if best_result:
        print(f"\n🏆 BEST CONFIGURATION:")
        for key, value in best_result.items():
            if key != "results":
                print(f"   {key}: {value}")
    
    print(f"\n📊 Results: {csv_file}, {json_file}")


if __name__ == "__main__":
    main()

