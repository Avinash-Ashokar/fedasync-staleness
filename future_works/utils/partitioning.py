import os
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from contextlib import redirect_stdout, redirect_stderr
import io


class DataDistributor:
    def __init__(self, dataset_name: str, data_dir: str = "./data"):
        """
        Flexible data distributor for federated learning experiments.

        Args:
            dataset_name (str): Name of dataset ('CIFAR10', 'MNIST', etc.)
            data_dir (str): Directory to store data.
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.train_dataset, self.test_dataset, self.num_classes = self._load_dataset()
        self.partitions = None

    def _load_dataset(self) -> Tuple[Any, Any, int]:
        """Load supported torchvision datasets."""
        transform = transforms.Compose([transforms.ToTensor()])
        buf = io.StringIO()

        if self.dataset_name == "cifar10":
            with redirect_stdout(buf), redirect_stderr(buf):
                train = datasets.CIFAR10(self.data_dir, train=True, download=True, transform=transform)
                test = datasets.CIFAR10(self.data_dir, train=False, download=True, transform=transform)
            num_classes = 10

        elif self.dataset_name == "mnist":
            with redirect_stdout(buf), redirect_stderr(buf):
                train = datasets.MNIST(self.data_dir, train=True, download=True, transform=transform)
                test = datasets.MNIST(self.data_dir, train=False, download=True, transform=transform)
            num_classes = 10

        elif self.dataset_name == "fashionmnist":
            with redirect_stdout(buf), redirect_stderr(buf):
                train = datasets.FashionMNIST(self.data_dir, train=True, download=True, transform=transform)
                test = datasets.FashionMNIST(self.data_dir, train=False, download=True, transform=transform)
            num_classes = 10

        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported yet.")

        return train, test, num_classes

    def distribute_data(self, num_clients: int, alpha: float = 0.5, seed: int = 42) -> Dict[int, List[int]]:
        """
        Perform Dirichlet-based data partitioning across clients (Non-IID).

        Args:
            num_clients (int): Number of clients.
            alpha (float): Dirichlet distribution parameter (smaller = more non-IID).
            seed (int): Random seed for reproducibility.
        """
        np.random.seed(seed)
        targets = np.array(self.train_dataset.targets)
        self.partitions = {i: [] for i in range(num_clients)}

        for cls in range(self.num_classes):
            idxs = np.where(targets == cls)[0]
            # Shuffle indices for this class
            np.random.shuffle(idxs)
            # Sample proportions from a Dirichlet distribution
            proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
            # Convert proportions to integer counts (floor) for each client
            int_props = np.floor(proportions * len(idxs)).astype(int)
            # Assign counts to clients
            start = 0
            for client_id, size in enumerate(int_props):
                self.partitions[client_id].extend(idxs[start:start + size])
                start += size
            # If any samples are left over due to floor truncation, assign them
            # to clients with the largest initial share (or random if equal).  This
            # ensures that the union of partitions covers the full dataset.
            remaining = len(idxs) - start
            if remaining > 0:
                # Rank clients by proportion (descending); break ties randomly
                ranked_clients = np.argsort(-proportions)
                # Distribute leftover samples in round‑robin order among ranked clients
                for i in range(remaining):
                    cid = ranked_clients[i % len(ranked_clients)]
                    self.partitions[int(cid)].append(idxs[start + i])

        for cid in self.partitions:
            np.random.shuffle(self.partitions[cid])

        return self.partitions

    # ... rest of partitioning.py remains unchanged ...


    def get_client_data(self, client_id: int) -> Subset:
        """
        Retrieve dataset subset for a specific client.

        Args:
            client_id (int): Client identifier.
        """
        if self.partitions is None:
            raise ValueError("Please run distribute_data() before accessing client data.")
        indices = self.partitions[client_id]
        return Subset(self.train_dataset, indices)

    def visualize_distribution(self, save_path: str = "./results/data_distribution_ieee.png") -> None:
        """
        Create IEEE-style stacked bar chart of sample counts per client.

        Args:
            save_path (str): File path to save the visualization.
        """
        if self.partitions is None:
            raise ValueError("Run distribute_data() before visualization.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        targets = np.array(self.train_dataset.targets)
        client_counts = np.zeros((len(self.partitions), self.num_classes), dtype=int)

        for cid, idxs in self.partitions.items():
            class_counts = np.bincount(targets[idxs], minlength=self.num_classes)
            client_counts[cid, :] = class_counts

        # IEEE single-column figure size (~3.5in wide)
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


if __name__ == "__main__":
    distributor = DataDistributor(dataset_name="CIFAR10")

    distributor.distribute_data(num_clients=21, alpha=1000, seed=42)
    distributor.visualize_distribution("./results/cifar10_distribution_ieee.png")

    # Retrieve client dataset subset
    client_data = distributor.get_client_data(0)
    print(f"✅ Client 0 has {len(client_data)} samples.")
