# async_fl_server.py
import torch
from copy import deepcopy
from utils.partitioning import DataDistributor
from .client import Client
import torch.nn.functional as F
import torch.nn as nn

# Simple CNN for CIFAR/MNIST/FashionMNIST
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)
        
        # Compute flattened feature size dynamically
        self._to_linear = None
        self.fc1 = nn.Linear(0, 128)  # placeholder
        self.fc2 = nn.Linear(128, num_classes)
        self._setup()

    def _setup(self):
        # Dummy forward pass to compute _to_linear
        x = torch.zeros(1, 3, 32, 32)  # CIFAR10 input size
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        self._to_linear = x.numel() // x.shape[0]
        self.fc1 = nn.Linear(self._to_linear, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class AsyncFLServer:
    """Federated Learning server simulating asynchronous updates."""
    def __init__(self, dataset_name="CIFAR10", num_clients=5, device="cpu"):
        self.device = device
        self.num_clients = num_clients
        self.distributor = DataDistributor(dataset_name)
        self.distributor.distribute_data(num_clients=num_clients)

        # Global model
        self.global_model = SimpleCNN()
        # Create client objects
        self.clients = [
            Client(i, deepcopy(self.global_model), self.distributor.get_client_data(i), device=device)
            for i in range(num_clients)
        ]

    def aggregate(self, client_params):
        """Aggregate client parameters using simple mean."""
        global_state = deepcopy(self.global_model.state_dict())
        for key in global_state.keys():
            global_state[key] = torch.stack([p[key] for p in client_params], dim=0).mean(0)
        self.global_model.load_state_dict(global_state)

    def run_async_round(self, epochs=1):
        """Run one asynchronous round (all clients train independently then aggregate)."""
        client_params = []
        for client in self.clients:
            client.train(epochs=epochs)
            client_params.append(client.get_parameters())
        self.aggregate(client_params)
        print("🌍 Async aggregation complete.")

    def visualize_distribution(self):
        """Visualize the data distribution."""
        self.distributor.visualize_distribution()
