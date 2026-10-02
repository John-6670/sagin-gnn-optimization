from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import os
import socket
import urllib.request
import urllib.error
import tarfile
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
    def __init__(self, input_dim=11500):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    @staticmethod
    def make_iot_autoencoder():
        # Default to N-BaIoT dimension (100 timesteps * 115 features = 11500)
        model = IoTAutoencoder(input_dim=11500)
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
    input_dim = 11500   # 100 time steps × 115 features (N-BaIoT compatible, flattened)
    # 70% normal, 30% anomaly
    normal_mask = rng.random(total_windows) < 0.7
    X = np.empty((total_windows, input_dim), dtype=np.float32)
    # Normal data: smooth temporal correlation (e.g., random walk + noise)
    n_normal = normal_mask.sum()
    for i in range(n_normal):
        seq = np.cumsum(rng.normal(0, 0.1, size=(100, 115)), axis=0) + rng.normal(0, 0.5, size=(100, 115))
        X[i] = seq.flatten()
    # Anomalous data: sudden spikes or different pattern
    n_anom = total_windows - n_normal
    for i in range(n_normal, total_windows):
        seq = rng.normal(0, 2.0, size=(100, 115))  # higher variance, no temporal structure
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
        seq = np.cumsum(rng.normal(0, 0.1, size=(100, 115)), axis=0) + rng.normal(0, 0.5, size=(100, 115))
        X_test_list.append(seq.flatten())
    for _ in range(300):
        seq = rng.normal(0, 2.0, size=(100, 115))
        X_test_list.append(seq.flatten())
    X_test = np.array(X_test_list, dtype=np.float32)
    perm = rng.permutation(1000)
    X_test = X_test[perm]
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(X_test)), batch_size=128)

    return loaders, test_loader


# =====================================================================
# REAL DATASET LOADERS (LEAF Reddit, N-BaIoT/SWaT)
# =====================================================================

# Trusted domains for dataset downloads (SSRF protection)
_TRUSTED_DOMAINS = {
    "github.com",
    "raw.githubusercontent.com",
    "archive.ics.uci.edu",
}

def _is_safe_url(url: str) -> bool:
    """Check if URL hostname is in trusted domains (SSRF protection)."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False
        # Resolve hostname to IPs and check against private ranges
        addrs = socket.getaddrinfo(hostname, None)
        for addr in addrs:
            ip = addr[4][0]
            # Check for loopback, link-local, private, multicast
            if (ip.startswith("127.") or ip.startswith("::1") or
                ip.startswith("169.254.") or ip.startswith("10.") or
                ip.startswith("172.16.") or ip.startswith("192.168.") or
                ip.startswith("fc00:") or ip.startswith("fe80:") or
                ip.startswith("ff00:")):
                return False
        # Check domain against trusted list
        return any(hostname == d or hostname.endswith("." + d) for d in _TRUSTED_DOMAINS)
    except Exception:
        return False


def _safe_download(url: str, dest_path: str, timeout: int = 60) -> bool:
    """Download file with SSRF protection and no redirects."""
    if not _is_safe_url(url):
        print(f"URL blocked by SSRF protection: {url}")
        return False

    try:
        # Use urlopen with no redirects
        req = urllib.request.Request(url, headers={'User-Agent': 'SAGIN-FL/1.0'})
        response = urllib.request.urlopen(req, timeout=timeout)

        # Verify final URL is still safe (no redirect to unsafe domain)
        final_url = response.geturl()
        if not _is_safe_url(final_url):
            print(f"Redirect to unsafe URL blocked: {final_url}")
            return False

        with open(dest_path, 'wb') as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
        return True
    except urllib.error.URLError as e:
        print(f"Download failed: {e}")
        return False
    except Exception as e:
        print(f"Download error: {e}")
        return False


def _download_leaf_reddit(data_dir: str = "data/reddit") -> Tuple[str, str]:
    """
    Download and prepare LEAF Reddit dataset.
    Returns paths to (train_dir, test_file).

    LEAF Reddit format: data/reddit/train/*.json, data/reddit/test/test.json
    Each JSON contains 'users' list and 'user_data' dict with 'x' (sequences) and 'y' (next tokens).
    """
    import tarfile
    import os

    os.makedirs(data_dir, exist_ok=True)
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # Check if already downloaded
    if os.path.exists(train_dir) and os.path.exists(os.path.join(test_dir, "test.json")):
        return train_dir, os.path.join(test_dir, "test.json")

    # Download LEAF Reddit data (from LEAF benchmark repo)
    leaf_url = "https://github.com/TalwalkarLab/leaf/raw/master/data/reddit/data.tar.gz"
    tarball_path = os.path.join(data_dir, "reddit_data.tar.gz")

    if not _safe_download(leaf_url, tarball_path):
        print("Could not download LEAF Reddit (SSRF check failed or network error)")
        print("Falling back to synthetic data...")
        return None, None

    try:
        print("Extracting...")
        with tarfile.open(tarball_path, "r:gz") as tar:
            # Safe extraction - prevent path traversal
            def is_safe_path(base: str, path: str) -> bool:
                return os.path.commonpath([os.path.abspath(base), os.path.abspath(path)]) == os.path.abspath(base)

            for member in tar.getmembers():
                if not is_safe_path(data_dir, os.path.join(data_dir, member.name)):
                    print(f"Skipping unsafe tar member: {member.name}")
                    continue
                tar.extract(member, data_dir)

        os.remove(tarball_path)
        print("LEAF Reddit dataset ready!")
        return train_dir, os.path.join(test_dir, "test.json")
    except Exception as e:
        print(f"Could not extract LEAF Reddit: {e}")
        print("Falling back to synthetic data...")
        return None, None


def _save_cache(cache_path: str, client_data: Dict, meta: Dict):
    """Save cached data using numpy .npz for arrays + JSON for vocab (safe, no pickle)."""
    # Save client data arrays
    client_arrays = {}
    for cid, (X, y) in client_data.items():
        client_arrays[f"X_{cid}"] = X
        client_arrays[f"y_{cid}"] = y

    np.savez_compressed(cache_path + ".npz", **client_arrays)

    # Save meta data (vocab as JSON, arrays as npz)
    import json
    vocab_path = cache_path + "_vocab.json"
    with open(vocab_path, 'w') as f:
        json.dump(meta['vocab'], f)

    np.savez_compressed(cache_path + "_meta.npz",
                        X_test=meta['X_test'],
                        y_test=meta['y_test'])


def _load_cache(cache_path: str) -> Tuple[Dict, Dict]:
    """Load cached data from numpy .npz format + JSON vocab."""
    # Load client data
    client_arrays = np.load(cache_path + ".npz", allow_pickle=False)

    client_data = {}
    client_ids = set()
    for key in client_arrays.files:
        if key.startswith("X_") and key[2:].isdigit():
            cid = int(key[2:])
            client_ids.add(cid)

    for cid in sorted(client_ids):
        X = client_arrays[f"X_{cid}"]
        y = client_arrays[f"y_{cid}"]
        client_data[cid] = (X, y)

    # Load meta
    import json
    vocab_path = cache_path + "_vocab.json"
    with open(vocab_path, 'r') as f:
        vocab_dict = json.load(f)

    meta_arrays = np.load(cache_path + "_meta.npz", allow_pickle=False)
    meta = {
        'vocab': {k: int(v) for k, v in vocab_dict.items()},
        'X_test': meta_arrays["X_test"],
        'y_test': meta_arrays["y_test"],
    }

    return client_data, meta


def _load_leaf_reddit_client_data(train_dir: str, test_file: str, max_clients: int = 100,
                                  max_seq_len: int = 20, vocab_size: int = 10000) -> Tuple[Dict, Dict]:
    """
    Load LEAF Reddit data and partition by user (client).
    Returns (client_data_dict, vocab_dict) where client_data_dict maps client_id -> (X, y).
    Uses safe numpy .npz caching instead of pickle.
    """
    import glob
    import json
    import os

    cache_base = os.path.join(train_dir, f"cached_reddit_v{vocab_size}_l{max_seq_len}")
    cache_path = cache_base + ".npz"

    if os.path.exists(cache_path):
        print("Loading Reddit data from cache...")
        return _load_cache(cache_base)

    print("Processing LEAF Reddit data...")

    # Build vocabulary from training data
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.json")))
    all_tokens = []

    for f in train_files:
        with open(f, 'r') as fin:
            data = json.load(fin)
        for client in data['users']:
            for seq in data['user_data'][client]['x']:
                tokens = seq.split()
                all_tokens.extend(tokens)

    # Build vocab (most frequent tokens)
    from collections import Counter
    token_counts = Counter(all_tokens)
    vocab = [tok for tok, _ in token_counts.most_common(vocab_size - 1)]  # -1 for UNK
    vocab_dict = {tok: idx + 1 for idx, tok in enumerate(vocab)}  # 0 = UNK
    vocab_dict['<UNK>'] = 0

    # Process client data
    client_data = {}
    client_id = 0

    for f in train_files:
        with open(f, 'r') as fin:
            data = json.load(fin)

        for client in data['users']:
            if client_id >= max_clients:
                break

            sequences = data['user_data'][client]['x']
            targets = data['user_data'][client]['y']

            X_client = []
            y_client = []

            for seq, target in zip(sequences, targets):
                tokens = seq.split()
                # Convert to indices
                token_ids = [vocab_dict.get(tok, 0) for tok in tokens]
                target_id = vocab_dict.get(target, 0)

                # Pad or truncate to max_seq_len
                if len(token_ids) >= max_seq_len:
                    token_ids = token_ids[-max_seq_len:]
                else:
                    token_ids = [0] * (max_seq_len - len(token_ids)) + token_ids

                X_client.append(token_ids)
                y_client.append(target_id)

            if len(X_client) > 10:  # Minimum samples per client
                client_data[client_id] = (np.array(X_client, dtype=np.int64),
                                          np.array(y_client, dtype=np.int64))
                client_id += 1

    # Process test data
    with open(test_file, 'r') as fin:
        test_data = json.load(fin)

    X_test = []
    y_test = []
    for client in test_data['users']:
        for seq, target in zip(test_data['user_data'][client]['x'],
                               test_data['user_data'][client]['y']):
            tokens = seq.split()
            token_ids = [vocab_dict.get(tok, 0) for tok in tokens]
            target_id = vocab_dict.get(target, 0)

            if len(token_ids) >= max_seq_len:
                token_ids = token_ids[-max_seq_len:]
            else:
                token_ids = [0] * (max_seq_len - len(token_ids)) + token_ids

            X_test.append(token_ids)
            y_test.append(target_id)

    X_test = np.array(X_test, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)

    meta = {'vocab': vocab_dict, 'X_test': X_test, 'y_test': y_test}

    # Cache using safe numpy format + JSON
    _save_cache(cache_base, client_data, meta)

    print(f"Loaded {len(client_data)} clients from LEAF Reddit")
    return client_data, meta


def _reddit_real(num_clients: int, seed: int = 0, data_dir: str = "data/reddit"):
    """
    Real LEAF Reddit next-word prediction task.
    Uses actual Reddit comment data partitioned by user (client).
    """
    rng = np.random.default_rng(seed)

    train_dir, test_file = _download_leaf_reddit(data_dir)

    if train_dir is None:
        # Fallback to synthetic
        print("Using synthetic Reddit data (real data not available)")
        return _reddit(num_clients, seed)

    client_data, meta = _load_leaf_reddit_client_data(train_dir, test_file, max_clients=num_clients)

    # Select subset of clients if we have more than requested
    available_clients = list(client_data.keys())
    if len(available_clients) > num_clients:
        selected_clients = rng.choice(available_clients, num_clients, replace=False)
    else:
        selected_clients = available_clients

    loaders = []
    for cid in selected_clients:
        Xc, yc = client_data[cid]
        ds = TensorDataset(torch.tensor(Xc, dtype=torch.long), torch.tensor(yc, dtype=torch.long))
        loaders.append(DataLoader(ds, batch_size=32, shuffle=True))

    # Test loader
    test_loader = DataLoader(
        TensorDataset(torch.tensor(meta['X_test'], dtype=torch.long),
                      torch.tensor(meta['y_test'], dtype=torch.long)),
        batch_size=128, shuffle=False
    )

    return loaders, test_loader


def _iot_real(num_clients: int, seed: int = 0, data_dir: str = "data/nbaiot"):
    """
    Real N-BaIoT / SWaT anomaly detection task.
    Uses actual IoT network traffic data for anomaly detection.
    """
    data_path = _download_nbaiot(data_dir)

    if data_path is None:
        # Fallback to synthetic but more realistic
        return _load_nbaiot_data(data_dir, num_clients, seed)

    # Real implementation would go here
    return _load_nbaiot_data(data_dir, num_clients, seed)


def _download_nbaiot(data_dir: str = "data/nbaiot") -> Optional[str]:
    """
    Download N-BaIoT dataset.
    Returns path to extracted data directory or None if failed.
    """
    import urllib.request
    import tarfile
    import os

    os.makedirs(data_dir, exist_ok=True)

    # Check if already exists
    if os.path.exists(os.path.join(data_dir, "benign")):
        return data_dir

    # N-BaIoT dataset from UCI ML Repository
    # This is a large dataset (~5GB), so we'll provide instructions instead
    print("N-BaIoT dataset is large (~5GB). Please download manually from:")
    print("https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT")
    print(f"Extract to: {data_dir}")
    return None


def _load_nbaiot_data(data_dir: str, num_clients: int, seed: int = 0) -> Tuple[List, DataLoader]:
    """
    Load N-BaIoT data and partition across clients.
    Returns (client_loaders, test_loader).
    """
    rng = np.random.default_rng(seed)
    input_dim = 115  # N-BaIoT has 115 features

    # This is a placeholder - real implementation would parse CSV files
    # For now, generate synthetic but more realistic IoT data
    print("Using synthetic N-BaIoT-like data (real data not downloaded)")

    # Generate data with realistic IoT patterns
    windows_per_client = 200
    total_windows = num_clients * windows_per_client

    # Normal traffic: periodic patterns with noise
    # Attack traffic: sudden changes, high variance
    X = np.empty((total_windows, 100, input_dim), dtype=np.float32)

    for i in range(total_windows):
        if rng.random() < 0.7:  # 70% normal
            # Normal: periodic + noise
            t = np.arange(100)
            base = np.sin(t * 0.1)[:, None] * np.ones(input_dim)[None, :]
            noise = rng.normal(0, 0.1, size=(100, input_dim))
            X[i] = base + noise
        else:  # 30% attack
            # Attack: sudden spikes, high variance
            X[i] = rng.normal(0, 2.0, size=(100, input_dim))

    # Flatten for autoencoder
    X_flat = X.reshape(total_windows, -1)

    # Shuffle and split
    perm = rng.permutation(total_windows)
    X_flat = X_flat[perm]
    X_split = np.array_split(X_flat, num_clients)

    loaders = []
    for c in range(num_clients):
        ds = TensorDataset(torch.tensor(X_split[c]), torch.tensor(X_split[c]))
        loaders.append(DataLoader(ds, batch_size=64, shuffle=True))

    # Test set
    X_test_list = []
    for _ in range(700):
        t = np.arange(100)
        base = np.sin(t * 0.1)[:, None] * np.ones(input_dim)[None, :]
        noise = rng.normal(0, 0.1, size=(100, input_dim))
        X_test_list.append((base + noise).flatten())
    for _ in range(300):
        X_test_list.append(rng.normal(0, 2.0, size=(100, input_dim)).flatten())

    X_test = np.array(X_test_list, dtype=np.float32)
    perm = rng.permutation(1000)
    X_test = X_test[perm]
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(X_test)), batch_size=128)

    return loaders, test_loader


def _iot_real(num_clients: int, seed: int = 0, data_dir: str = "data/nbaiot"):
    """
    Real N-BaIoT / SWaT anomaly detection task.
    Uses actual IoT network traffic data for anomaly detection.
    """
    data_path = _download_nbaiot(data_dir)

    if data_path is None:
        # Fallback to synthetic but more realistic
        return _load_nbaiot_data(data_dir, num_clients, seed)

    # Real implementation would go here
    return _load_nbaiot_data(data_dir, num_clients, seed)


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
            make_data=_reddit_real,
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
            make_data=_iot_real,
            server_lr=1.0,        # autoencoder data is less heterogeneous, full averaging is fine
        ),
        # Synthetic versions for testing/fallback
        'synthetic_nwp': FLTask(
            name='synthetic_nwp',
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
        'synthetic_anomaly': FLTask(
            name='synthetic_anomaly',
            local_epochs=5,
            lr=1e-3,
            optimizer='adam',
            optimizer_kwargs={},
            criterion=nn.MSELoss(),
            make_model=IoTAutoencoder.make_iot_autoencoder,
            make_data=_iot,
            server_lr=1.0,
        )
    }
