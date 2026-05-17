"""
Baseline federated learning algorithm: FedAvg on non-IID CIFAR-10.

FedAvg (McMahan et al., AISTATS 2017) is the canonical federated averaging
algorithm. Clients train locally with SGD and the server averages their
model parameters weighted by dataset size.

Dataset: CIFAR-10, 20 clients, Dirichlet alpha=0.1 (highly non-IID)
Model: FedAvgCNN (4-layer CNN, ~100K parameters)

Agents should modify this file to improve federated learning:
- Better aggregation: FedProx proximal term, SCAFFOLD variance reduction,
  FedDyn dynamic regularization, MOON contrastive learning
- Better local training: lr scheduling, multiple local epochs, regularization
- Model architecture: wider/deeper CNN, ResNet, normalization layers
- Adaptive aggregation: momentum, partial model aggregation
"""
import copy
import torch
import torch.nn as nn
import numpy as np


def get_fl_config():
    """Return federated learning hyperparameters."""
    return {
        'global_rounds': 100,
        'local_epochs': 1,
        'lr': 0.005,
        'batch_size': 10,
        'join_ratio': 1.0,
        'num_clients': 20,
        'num_classes': 10,
    }


class FedAvgCNN(nn.Module):
    """4-layer CNN for CIFAR-10 federated learning.
    Same architecture as PFLlib's FedAvgCNN(in_features=3, num_classes=10, dim=1600).
    """
    def __init__(self, in_features=3, num_classes=10, dim=1600):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


def get_model(num_classes=10):
    """Return the model for federated training."""
    return FedAvgCNN(in_features=3, num_classes=num_classes, dim=1600)


class FLClient:
    """Federated learning client (local training)."""

    def __init__(self, client_id, train_data, device='cuda'):
        self.client_id = client_id
        self.train_data = train_data
        self.device = device
        self.model = None

    def set_model(self, global_model_state):
        """Receive global model parameters."""
        if self.model is None:
            self.model = get_model().to(self.device)
        self.model.load_state_dict(copy.deepcopy(global_model_state))

    def local_train(self):
        """Run local training and return updated model state dict + sample count."""
        config = get_fl_config()
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=config['lr'])
        criterion = nn.CrossEntropyLoss()

        loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=config['batch_size'],
            shuffle=True, drop_last=False
        )

        for epoch in range(config['local_epochs']):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

        return self.model.state_dict(), len(self.train_data)


class FLServer:
    """Federated learning server (aggregation)."""

    def __init__(self, device='cuda'):
        self.device = device
        self.global_model = get_model().to(device)

    def get_global_state(self):
        return self.global_model.state_dict()

    def aggregate(self, client_states, client_weights):
        """FedAvg: weighted average of client model parameters."""
        total_weight = sum(client_weights)
        global_state = self.global_model.state_dict()

        for key in global_state:
            global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
            for state, weight in zip(client_states, client_weights):
                global_state[key] += state[key].float() * (weight / total_weight)

        self.global_model.load_state_dict(global_state)

    def evaluate(self, test_data_list):
        """Evaluate global model on each client's test data.
        Returns list of per-client accuracies."""
        self.global_model.eval()
        client_accs = []

        with torch.no_grad():
            for test_data in test_data_list:
                if len(test_data) == 0:
                    continue
                loader = torch.utils.data.DataLoader(
                    test_data, batch_size=64, shuffle=False
                )
                correct = 0
                total = 0
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = self.global_model(x)
                    _, predicted = output.max(1)
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()
                if total > 0:
                    client_accs.append(correct / total)

        return client_accs
