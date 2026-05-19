from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models

from .device import DEVICE


def make_resnet18_cifar10(num_classes=10):
    model = models.resnet18(weights=None)  # no pretraining
    # Replace first convolutional layer
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove the early max‑pool (which down‑samples 4× → we keep spatial size)
    model.maxpool = nn.Identity()
    # Replace final FC layer
    model.fc = nn.Linear(512, num_classes)
    return model.to(DEVICE)


class RedditLSTM(nn.Module):
    def __init__(self, vocab_size=2260, emb_dim=128, hidden_dim=512, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        return self.fc(out[:, -1, :])
    
    @staticmethod
    def make_reddit_lstm():
        model = RedditLSTM()
        return model.to(DEVICE)  


class IoTAutoencoder(nn.Module):
    def __init__(self, input_dim=8100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    @staticmethod
    def make_iot_autoencoder():
        model = IoTAutoencoder()
        return model.to(DEVICE)  
    

@dataclass
class FLTask:
    name: str
    local_epochs: int
    lr: float
    optimizer: str           # 'sgd' or 'adam'
    optimizer_kwargs: Dict   # momentum, weight_decay, etc.
    criterion: nn.Module
    make_model: callable
    make_data: callable
    grad_clip: float = None
    
    def get_model(self):
        return self.make_model()
    
    def get_data_loaders(self, num_clients, seed=0):
        return self.make_data(num_clients, seed)


def _split(y, n, a, rng):
    idx = np.arange(len(y))
    out = [[] for _ in range(n)]
    for c in np.unique(y):
        cidx = idx[y == c]
        rng.shuffle(cidx)
        p = rng.dirichlet([a]*n)
        cuts = (np.cumsum(p)*len(cidx)).astype(int)[:-1]
        
        for i, s in enumerate(np.split(cidx, cuts)):
            out[i].extend(s.tolist())
    
    return [np.array(x, dtype=int) for x in out]


def _cifar(num_clients, seed=0):
    rng = np.random.default_rng(seed)
    # Total samples = num_clients * 500
    n_total = num_clients * 500
    X = rng.normal(size=(n_total, 3, 32, 32)).astype(np.float32)
    y = rng.integers(0, 10, n_total)

    # Dirichlet split
    client_indices = _split(y, num_clients, 0.5, rng)

    # 20% symmetric label noise
    for cid in range(num_clients):
        idx = client_indices[cid]
        labels = y[idx]
        flip = rng.random(len(idx)) < 0.2
        labels[flip] = rng.integers(0, 10, size=flip.sum())
        y[idx] = labels

    # Force exactly 500 samples per client (truncate/duplicate)
    for cid in range(num_clients):
        idx = client_indices[cid]
        if len(idx) > 500:
            client_indices[cid] = rng.choice(idx, 500, replace=False)
        elif len(idx) < 500:
            extra = rng.choice(idx, 500 - len(idx), replace=True)
            client_indices[cid] = np.concatenate([idx, extra])

    # DataLoaders (batch size 32)
    loaders = []
    for cid in range(num_clients):
        idx = client_indices[cid]
        ds = TensorDataset(torch.tensor(X[idx]), torch.tensor(y[idx], dtype=torch.long))
        loaders.append(DataLoader(ds, batch_size=32, shuffle=True))

    # Test set
    X_test = rng.normal(size=(10000, 3, 32, 32)).astype(np.float32)
    y_test = rng.integers(0, 10, 10000)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long)), batch_size=128)

    return loaders, test_loader


def _reddit(num_clients, seed=0):
    rng = np.random.default_rng(seed)
    vocab_size = 2260  # matches model
    # Total number of sequences = sum of per‑client sequence counts
    # Clients’ seq counts follow power law: p(k) ∝ k^{-γ}, γ = 2.0 for instance
    # generate integer counts between 100 and 1000
    min_seq, max_seq = 100, 1000
    # Sample raw counts from Pareto distribution, then clip
    raw = (rng.pareto(2.0, size=num_clients) * 100).astype(int) + 50
    seq_counts = np.clip(raw, min_seq, max_seq)
    total_seqs = seq_counts.sum()

    # Max sequence length = 20 (table)
    max_len = 20
    # Generate text data: X (total_seqs, max_len), y next word
    X = rng.integers(1, vocab_size, size=(total_seqs, max_len))  # 0 reserved for padding?
    y = rng.integers(1, vocab_size, size=total_seqs)

    # Split among clients
    client_data = []
    start = 0
    for c in range(num_clients):
        n = seq_counts[c]
        end = start + n
        client_data.append((X[start:end], y[start:end]))
        start = end

    loaders = []
    for Xc, yc in client_data:
        ds = TensorDataset(torch.tensor(Xc, dtype=torch.long), torch.tensor(yc, dtype=torch.long))
        loaders.append(DataLoader(ds, batch_size=20, shuffle=True))   # batch size 20

    # Test set
    X_test = rng.integers(1, vocab_size, size=(2000, max_len))
    y_test = rng.integers(1, vocab_size, size=2000)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)), batch_size=128)

    return loaders, test_loader


def _iot(num_clients, seed=0):
    rng = np.random.default_rng(seed)
    windows_per_client = 200
    total_windows = num_clients * windows_per_client
    input_dim = 8100   # 100 time steps × 81 features (flattened)
    # 70% normal, 30% anomaly
    normal_mask = rng.random(total_windows) < 0.7
    X = np.empty((total_windows, input_dim), dtype=np.float32)
    # Normal data: smooth temporal correlation (e.g., random walk + noise)
    n_normal = normal_mask.sum()
    for i in range(n_normal):
        seq = np.cumsum(rng.normal(0, 0.1, size=(100, 81)), axis=0) + rng.normal(0, 0.5, size=(100, 81))
        X[i] = seq.flatten()
    # Anomalous data: sudden spikes or different pattern
    n_anom = total_windows - n_normal
    for i in range(n_normal, total_windows):
        seq = rng.normal(0, 2.0, size=(100, 81))  # higher variance, no temporal structure
        X[i] = seq.flatten()

    # Shuffle
    perm = rng.permutation(total_windows)
    X = X[perm]
    # Split equally across clients
    X_split = np.array_split(X, num_clients)

    loaders = []
    for c in range(num_clients):
        ds = TensorDataset(torch.tensor(X_split[c]), torch.tensor(X_split[c]))  # reconstruction target = input
        loaders.append(DataLoader(ds, batch_size=64, shuffle=True))   # batch size 64

    # Test set: similar generation
    X_test = rng.normal(0, 1, size=(1000, input_dim)).astype(np.float32)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(X_test)), batch_size=128)

    return loaders, test_loader


def get_task_registry() -> Dict[str, FLTask]:
    return {
        'cifar10': FLTask(
            name='cifar10_resnet_dirichlet',
            local_epochs=5,
            lr=0.01,
            optimizer='sgd',
            optimizer_kwargs={'momentum': 0.9},
            criterion=nn.CrossEntropyLoss(),
            make_model=lambda: make_resnet18_cifar10(10),
            make_data=_cifar
        ),
        'reddit_nwp': FLTask(
            name='reddit_nwp',
            local_epochs=3,
            lr=0.5,
            optimizer='sgd',
            optimizer_kwargs={},
            criterion=nn.CrossEntropyLoss(),
            make_model=RedditLSTM.make_reddit_lstm,
            make_data=_reddit,
            grad_clip=1.0
        ),
        'iot_anomaly': FLTask(
            name='iot_anomaly',
            local_epochs=10,
            lr=1e-3,
            optimizer='adam',
            optimizer_kwargs={},
            criterion=nn.MSELoss(),
            make_model=IoTAutoencoder.make_iot_autoencoder,
            make_data=_iot    
        )
    }
