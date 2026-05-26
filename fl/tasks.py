from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

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
    server_lr: float = 1.0   # scales how far the global model moves toward mean(local models)
    
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
    train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    y = np.array(train_ds.targets)
    client_indices = _split(y, num_clients, 0.5, rng)

    # Cap each client at 500 samples
    for cid in range(num_clients):
        idx = client_indices[cid]
        if len(idx) > 500:
            client_indices[cid] = rng.choice(idx, 500, replace=False)
        elif len(idx) < 500:
            extra = rng.choice(idx, 500 - len(idx), replace=True)
            client_indices[cid] = np.concatenate([idx, extra])

    loaders = []
    for cid in range(num_clients):
        loaders.append(DataLoader(Subset(train_ds, client_indices[cid].tolist()),
                                  batch_size=32, shuffle=True))

    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    return loaders, test_loader


def _make_reddit_sequences(rng, n_seqs, client_topic, n_topics, vocab_size, max_len, tokens_per_topic):
    """
    Structured synthetic language data.  Each client belongs to one of n_topics
    "communities".  Within a community, tokens follow a Markov chain: with
    probability 0.75 the next token stays in the same topic-vocabulary, so the
    last token in the context is predictive of the target.
    """
    topic_start = client_topic * tokens_per_topic
    topic_end = topic_start + tokens_per_topic
    Xc, yc = [], []
    tok = int(rng.integers(topic_start, topic_end))
    for _ in range(n_seqs):
        seq = [tok]
        for _ in range(max_len):
            if rng.random() < 0.75:
                tok = int(rng.integers(topic_start, topic_end))
            else:
                tok = int(rng.integers(1, vocab_size))
            seq.append(tok)
        Xc.append(seq[:max_len])
        yc.append(seq[max_len])       # next-word target is predictable
    return np.array(Xc, dtype=np.int64), np.array(yc, dtype=np.int64)


def _reddit(num_clients, seed=0):
    rng = np.random.default_rng(seed)
    vocab_size = 2260
    max_len = 20
    n_topics = 10
    tokens_per_topic = vocab_size // n_topics  # 226 tokens per topic

    # Assign each client a primary topic (Dirichlet-like heterogeneity)
    client_topics = rng.integers(0, n_topics, size=num_clients)

    # Per-client sequence counts (Pareto power law)
    raw = (rng.pareto(2.0, size=num_clients) * 100).astype(int) + 50
    seq_counts = np.clip(raw, 100, 1000)

    loaders = []
    for c in range(num_clients):
        Xc, yc = _make_reddit_sequences(
            rng, int(seq_counts[c]), int(client_topics[c]),
            n_topics, vocab_size, max_len, tokens_per_topic
        )
        ds = TensorDataset(torch.tensor(Xc, dtype=torch.long), torch.tensor(yc, dtype=torch.long))
        loaders.append(DataLoader(ds, batch_size=20, shuffle=True))

    # Test set: balanced across all topics so accuracy measures real generalisation
    X_test, y_test = [], []
    for topic in range(n_topics):
        Xb, yb = _make_reddit_sequences(rng, 200, topic, n_topics, vocab_size, max_len, tokens_per_topic)
        X_test.append(Xb)
        y_test.append(yb)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)),
        batch_size=128
    )
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

    # Test set: same mix as training (70 % normal random-walk, 30 % spike anomaly)
    # so reconstruction loss reflects what the model actually learned.
    X_test_list = []
    for _ in range(700):
        seq = np.cumsum(rng.normal(0, 0.1, size=(100, 81)), axis=0) + rng.normal(0, 0.5, size=(100, 81))
        X_test_list.append(seq.flatten())
    for _ in range(300):
        seq = rng.normal(0, 2.0, size=(100, 81))
        X_test_list.append(seq.flatten())
    X_test = np.array(X_test_list, dtype=np.float32)
    perm = rng.permutation(1000)
    X_test = X_test[perm]
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(X_test)), batch_size=128)

    return loaders, test_loader


def get_task_registry() -> Dict[str, FLTask]:
    return {
        'cifar10': FLTask(
            name='cifar10_resnet_dirichlet',
            local_epochs=2,       # fewer epochs → less client drift with heterogeneous data
            lr=0.01,
            optimizer='sgd',
            optimizer_kwargs={'momentum': 0.9},
            criterion=nn.CrossEntropyLoss(),
            make_model=lambda: make_resnet18_cifar10(10),
            make_data=_cifar,
            server_lr=0.5,        # damp global update to prevent divergence
        ),
        'reddit_nwp': FLTask(
            name='reddit_nwp',
            local_epochs=3,
            lr=0.3,
            optimizer='sgd',
            optimizer_kwargs={},
            criterion=nn.CrossEntropyLoss(),
            make_model=RedditLSTM.make_reddit_lstm,
            make_data=_reddit,
            grad_clip=1.0,
            server_lr=1.0,
        ),
        'iot_anomaly': FLTask(
            name='iot_anomaly',
            local_epochs=5,
            lr=1e-3,
            optimizer='adam',
            optimizer_kwargs={},
            criterion=nn.MSELoss(),
            make_model=IoTAutoencoder.make_iot_autoencoder,
            make_data=_iot,
            server_lr=1.0,        # autoencoder data is less heterogeneous, full averaging is fine
        )
    }
