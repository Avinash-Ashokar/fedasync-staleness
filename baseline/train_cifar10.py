import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import torch
except Exception as e:
    sys.stderr.write('ERROR: PyTorch not installed or incompatible with this interpreter.\n')
    sys.stderr.write(f'{e.__class__.__name__}: {e}\n')
    sys.stderr.write('Hint: activate .venv_311 and pip install torch torchvision numpy pyyaml.\n')
    sys.exit(1)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.helper import get_device, set_seed
from utils.model import build_resnet18

print(f"[env] python={sys.version.split()[0]} torch={torch.__version__} mps={getattr(torch.backends.mps,'is_available',lambda:False)()} cuda={torch.cuda.is_available()}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std), transforms.RandomErasing(p=0.25), transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)])
    val_test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_full = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=None)
    test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=val_test_tf)

    indices = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_indices, val_indices = indices[:int(0.8*len(train_full))], indices[int(0.8*len(train_full)):]

    class SubsetDataset:
        def __init__(self, d, i, t): self.d, self.i, self.t = d, i, t
        def __len__(self): return len(self.i)
        def __getitem__(self, idx): img, tgt = self.d[self.i[idx]]; return (self.t(img), tgt) if self.t else (img, tgt)
    train_dataset = SubsetDataset(train_full, train_indices, train_tf)
    val_dataset = SubsetDataset(train_full, val_indices, val_test_tf)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    model = build_resnet18(num_classes=10, pretrained=False)
    model, criterion = model.to(device), nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

    logs_base = Path("logs") / "avinash" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_baseline"
    logs_base.mkdir(parents=True, exist_ok=True)
    csv_header = "time_sec,epoch,train_loss,val_acc,test_acc,lr,wd,batch,seed"
    with open(logs_base / "COMMIT.txt", "w") as f:
        f.write(f"{subprocess.check_output(['git','rev-parse','--short','HEAD'],text=True).strip()},{csv_header}\n")
    csv.writer(open(logs_base / "metrics.csv", "w", newline="")).writerow(csv_header.split(","))

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum, train_count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x.size(0)
            train_count += x.size(0)
        train_loss = train_loss_sum / train_count

        def eval_acc(loader):
            c, t = 0, 0
            with torch.no_grad():
                for x, y in loader:
                    c += (model(x.to(device)).argmax(1) == y.to(device)).sum().item()
                    t += y.size(0)
            return c / t

        model.eval()
        val_acc, test_acc = eval_acc(val_loader), eval_acc(test_loader)
        with open(logs_base / "metrics.csv", "a", newline="") as f:
            csv.writer(f).writerow([f"{time.time()-start_time:.3f}", epoch, f"{train_loss:.6f}", f"{val_acc:.6f}", f"{test_acc:.6f}", f"{args.lr:.6f}", f"{args.wd:.6f}", args.batch, args.seed])
        print(f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.6f} val_acc={val_acc:.6f} test_acc={test_acc:.6f}", flush=True)
        scheduler.step()

if __name__ == "__main__": main()
