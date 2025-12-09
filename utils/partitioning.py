import os
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any


class DataDistributor:
    def __init__(self, dataset_name: str, data_dir: str = "./data"):
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.train_dataset, self.test_dataset, self.num_classes = self._load_dataset()
        self.partitions = None

    def _load_dataset(self) -> Tuple[Any, Any, int]:
        if self.dataset_name == "cifar10":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ])
            train = datasets.CIFAR10(self.data_dir, train=True, download=True, transform=transform_train)
            test = datasets.CIFAR10(self.data_dir, train=False, download=True, transform=transform_test)
            num_classes = 10

        elif self.dataset_name == "mnist":
            transform = transforms.Compose([transforms.ToTensor()])
            train = datasets.MNIST(self.data_dir, train=True, download=True, transform=transform)
            test = datasets.MNIST(self.data_dir, train=False, download=True, transform=transform)
            num_classes = 10

        elif self.dataset_name == "fashionmnist":
            transform = transforms.Compose([transforms.ToTensor()])
            train = datasets.FashionMNIST(self.data_dir, train=True, download=True, transform=transform)
            test = datasets.FashionMNIST(self.data_dir, train=False, download=True, transform=transform)
            num_classes = 10

        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported yet.")

        return train, test, num_classes

    def distribute_data(self, num_clients: int, alpha: float = 0.5, seed: int = 42) -> Dict[int, List[int]]:
        np.random.seed(seed)
        targets = np.array(self.train_dataset.targets)
        self.partitions = {i: [] for i in range(num_clients)}

        for cls in range(self.num_classes):
            idxs = np.where(targets == cls)[0]
            np.random.shuffle(idxs)
            proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
            proportions = np.array([p * len(idxs) for p in proportions]).astype(int)

            start = 0
            for client_id, size in enumerate(proportions):
                self.partitions[client_id].extend(idxs[start:start + size])
                start += size

            total_assigned = proportions.sum()
            leftover = len(idxs) - total_assigned
            if leftover > 0:
                recipients = np.random.choice(num_clients, size=leftover, replace=True)
                for idx, r in enumerate(recipients):
                    self.partitions[r].append(idxs[total_assigned + idx])

        for cid in self.partitions:
            np.random.shuffle(self.partitions[cid])

        return self.partitions

    def get_client_data(self, client_id: int) -> Subset:
        if self.partitions is None:
            raise ValueError("Please run distribute_data() before accessing client data.")
        indices = self.partitions[client_id]
        return Subset(self.train_dataset, indices)

    def visualize_distribution(self, save_path: str = "./results/data_distribution_ieee.png") -> None:
        if self.partitions is None:
            raise ValueError("Run distribute_data() before visualization.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        targets = np.array(self.train_dataset.targets)
        client_counts = np.zeros((len(self.partitions), self.num_classes), dtype=int)

        for cid, idxs in self.partitions.items():
            class_counts = np.bincount(targets[idxs], minlength=self.num_classes)
            client_counts[cid, :] = class_counts

        fig, ax = plt.subplots(figsize=(1.8, 1.2), dpi=300)
        bottom = np.zeros(len(self.partitions))
        colors = plt.get_cmap("tab20").colors

        for cls in range(self.num_classes):
            ax.bar(
                x=np.arange(len(self.partitions)),
                height=client_counts[:, cls],
                bottom=bottom,
                color=colors[cls % len(colors)],
                linewidth=0.1,
                edgecolor="white",
            )
            bottom += client_counts[:, cls]

        ax.set_xlabel("Client ID", fontsize=8)
        ax.set_ylabel("Samples", fontsize=8)
        ax.set_title(f"{self.dataset_name.upper()} Data Distribution Among Clients", fontsize=9)
        ax.tick_params(axis="both", which="major", labelsize=7)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"📊 Data distribution plot saved at: {save_path}")
