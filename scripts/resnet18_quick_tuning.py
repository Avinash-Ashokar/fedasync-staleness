import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import numpy as np
import csv
import json
from pathlib import Path
from datetime import datetime
import os

# Ensure PyTorch is installed and device is available
try:
    import torch
    print(f"✅ PyTorch is available")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Device available: {device}")
except ImportError:
    print("❌ PyTorch is not installed. Please install it: pip install torch torchvision torchaudio")
    exit()

# Set random seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data loading and transformations
def get_cifar10_dataloaders(data_dir: str, batch_size: int):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

# Import ResNet-18 from utils
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from utils.model import build_resnet18

# Training function
def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, config: dict, device: torch.device):
    set_seed(config["seed"])
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    best_test_acc = 0.0
    best_epoch = 0
    
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        test_loss /= total_test
        test_acc = correct_test / total_test

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1

    return best_test_acc, best_epoch, test_acc, train_acc

if __name__ == "__main__":
    # Hyperparameter grid (smaller for quick test)
    lrs = [0.001, 0.01]
    batch_sizes = [64, 128]
    weight_decays = [0.0, 1e-4]
    optimizers = ["adam", "sgd"]
    epochs = 30  # Reduced epochs for quick test
    
    configs = []
    for lr in lrs:
        for bs in batch_sizes:
            for wd in weight_decays:
                for opt in optimizers:
                    configs.append({
                        "lr": lr,
                        "batch_size": bs,
                        "weight_decay": wd,
                        "optimizer": opt,
                        "epochs": epochs,
                        "momentum": 0.9 if opt == "sgd" else 0.0,
                        "seed": 42,
                    })
    
    total = len(configs)
    print(f"\nTotal experiments: {total}")
    
    # Create output directory
    output_dir = Path("hyperparameter_tuning_results/resnet18")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = output_dir / f"quick_results_{timestamp}.csv"
    json_file = output_dir / f"quick_summary_{timestamp}.json"
    
    # Write CSV header
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lr", "batch_size", "weight_decay", "optimizer", "momentum",
            "best_test_acc", "best_epoch", "final_test_acc", "final_train_acc"
        ])
    
    results = []
    best_result = None
    best_acc = 0.0
    
    # Run experiments
    for i, config in enumerate(configs, 1):
        print(f"Running quick experiment {i}/{total}: {config}")
        model = build_resnet18(num_classes=10)
        train_loader, test_loader = get_cifar10_dataloaders("./data", config["batch_size"])
        
        best_test_acc, best_epoch, final_test_acc, final_train_acc = train_model(
            model, train_loader, test_loader, config, device
        )
        
        result = {
            "lr": config["lr"],
            "batch_size": config["batch_size"],
            "weight_decay": config["weight_decay"],
            "optimizer": config["optimizer"],
            "momentum": config["momentum"],
            "best_test_acc": round(best_test_acc * 100, 4),
            "best_epoch": best_epoch,
            "final_test_acc": round(final_test_acc * 100, 4),
            "final_train_acc": round(final_train_acc * 100, 4),
        }
        results.append(result)
        
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                result["lr"], result["batch_size"], result["weight_decay"],
                result["optimizer"], result["momentum"], result["best_test_acc"],
                result["best_epoch"], result["final_test_acc"], result["final_train_acc"]
            ])
        
        if result["best_test_acc"] > best_acc:
            best_acc = result["best_test_acc"]
            best_result = result
    
    print("\n--- Quick Hyperparameter Tuning Complete ---")
    print(f"Results saved to: {csv_file}")
    print(f"Best configuration found: {best_result}")
    
    with open(json_file, "w") as f:
        json.dump(best_result, f, indent=4)
    print(f"Summary saved to: {json_file}")

