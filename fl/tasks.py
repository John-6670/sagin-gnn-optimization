from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ResNet18Like(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)
    

class RedditLSTM(nn.Module):
    def __init__(self, vocab=5000, emb=128, h=128):
        super().__init__()
        self.e = nn.Embedding(vocab, emb)
        self.l = nn.LSTM(emb, h, 2, batch_first=True)
        self.f = nn.Linear(h, vocab)
    
    def forward(self, x):
        y,_ = self.l(self.e(x))
        return self.f(y[:, -1, :])
    

class IoTAE(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.n = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
    
    def forward(self, x):
        return self.n(x)
    

@dataclass
class FLTask:
    name:str
    local_epochs:int
    lr:float
    criterion:nn.Module
    make_model:callable
    make_data:callable
    
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


def _cifar(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(10000, 3, 32, 32)).astype(np.float32)
    y = rng.integers(0, 10, 10000)
    sp = _split(y, n, 0.5, rng)
    
    L = [DataLoader(
            TensorDataset(torch.tensor(X[s]), torch.tensor(y[s], dtype=torch.long)),
            batch_size=64, shuffle=True
        ) for s in sp]
    T = DataLoader(
            TensorDataset(torch.tensor(X[:2000]), torch.tensor(y[:2000], dtype=torch.long)),
            batch_size=128
        )
    
    return L, T


def _reddit(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 5000, (12000, 30))
    y = rng.integers(0, 5000, 12000)
    sp = np.array_split(np.arange(len(X)), n)
    
    L = [DataLoader(
            TensorDataset(torch.tensor(X[s], dtype=torch.long), torch.tensor(y[s], dtype=torch.long)),
            batch_size=64,shuffle=True
        ) for s in sp]
    T = DataLoader(
            TensorDataset(torch.tensor(X[:2000], dtype=torch.long), torch.tensor(y[:2000], dtype=torch.long)),
            batch_size=128
        )
    
    return L, T


def _iot(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(8000, 128)).astype(np.float32)
    sp = np.array_split(np.arange(len(X)), n)
    
    L = [DataLoader(
            TensorDataset(torch.tensor(X[s]), torch.tensor(X[s])),
            batch_size=64,shuffle=True
        ) for s in sp]
    T = DataLoader(
            TensorDataset(torch.tensor(X[:1000]), torch.tensor(X[:1000])),
            batch_size=128
        )
    
    return L, T


def get_task_registry() -> Dict[str, FLTask]:
    return {
        'cifar10': FLTask('cifar10', 1, 1e-2, nn.CrossEntropyLoss(), lambda:ResNet18Like(10), _cifar),
        'reddit_nwp': FLTask('reddit_nwp', 1, 5e-3, nn.CrossEntropyLoss(), lambda:RedditLSTM(), _reddit),
        'iot_anomaly': FLTask('iot_anomaly', 1, 1e-3, nn.MSELoss(), lambda:IoTAE(128), _iot)
    }
