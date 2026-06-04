from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import HeteroData

from optimization.objective import compute_utility
from simulation.topology.nodes import NodeType, generate_nodes


class SAGINSnapshotDataset(Dataset):
    def __init__(self, num_samples=10000, area_size=2000, gradient_dim=100, snr_threshold=1e-6, lhs_seed=42):
        self.num_samples = num_samples
        self.area_size = area_size
        self.gradient_dim = gradient_dim
        self.snr_threshold = snr_threshold
        self.rng = np.random.default_rng(lhs_seed)
        self.lhs = self._latin_hypercube(num_samples, 4)
        self.cache = [None] * num_samples

    def _latin_hypercube(self, n, d):
        cut = np.linspace(0, 1, n + 1)
        u = self.rng.uniform(size=(n, d))
        a, b = cut[:-1], cut[1:]
        pts = u * (b - a)[:, None] + a[:, None]
        for j in range(d):
            self.rng.shuffle(pts[:, j])
        return pts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.cache[idx] is not None:
            return self.cache[idx]
        
        p = self.lhs[idx]
        clients_n = 125
        sats = 9
        uavs = 1
        ground = 3
        nodes = generate_nodes(sats, uavs, ground, clients_n, self.area_size, self.gradient_dim)
        clients = [n for n in nodes if n.type == NodeType.CLIENT]
        candidates = [n for n in nodes if n.type != NodeType.CLIENT]

        # Build graph data (same as before)
        data = HeteroData()
        type_map = {NodeType.SATELLITE: 'satellite', NodeType.UAV: 'hap', 
                    NodeType.GROUND: 'ground', NodeType.CLIENT: 'client'}
        node_to_local = {}
        node_objects = {t: [] for t in type_map.values()}

        for n in nodes:
            tname = type_map[n.type]
            node_objects[tname].append(n)
            type_feat = {
                NodeType.SATELLITE: [1,0,0,0],
                NodeType.UAV: [0,1,0,0],
                NodeType.GROUND: [0,0,1,0],
                NodeType.CLIENT: [0,0,0,1],
            }[n.type]

            feat = np.array([*n.position, n.power, n.noise_variance, *type_feat], dtype=np.float32)
            node_to_local[n.id] = (tname, len(data[tname].x) if hasattr(data, tname) and data[tname].x else 0)
            if not hasattr(data, tname) or not data[tname].x:
                data[tname].x = []
            data[tname].x.append(feat)

        for t in type_map.values():
            if hasattr(data[t], 'x') and data[t].x:
                data[t].x = torch.tensor(np.array(data[t].x), dtype=torch.float32)

        # === PAPER-ALIGNED BASE SELECTION (diverse tiers) ===
        selected = []
        for tier in [NodeType.GROUND, NodeType.UAV, NodeType.SATELLITE]:
            tier_nodes = [n for n in candidates if n.type == tier]
            if tier_nodes:
                selected.append(tier_nodes[self.rng.integers(len(tier_nodes))])

        edge_store = {}
        edge_attr_store = {}
        for c in clients:
            for s in candidates:
                snr = c.compute_snr_to(s)

                if (
                    not np.isfinite(snr)
                    or snr < self.snr_threshold
                ):
                    continue

                if c.id == s.id:
                    continue

                src_t, src_i = node_to_local[c.id]
                dst_t, dst_i = node_to_local[s.id]

                # Forward edge:
                # client -> server
                et = (src_t, 'link', dst_t)
                edge_store.setdefault(et, [[], []])
                edge_store[et][0].append(src_i)
                edge_store[et][1].append(dst_i)

                edge_attr_store.setdefault(et, [])
                edge_attr_store[et].append([float(snr)])

                # Reverse edge:
                # server -> client
                rev_et = (dst_t, 'rev_link', src_t)
                edge_store.setdefault(rev_et, [[], []])
                edge_store[rev_et][0].append(dst_i)
                edge_store[rev_et][1].append(src_i)

                edge_attr_store.setdefault(rev_et, [])
                edge_attr_store[rev_et].append([float(snr)])

        for et, eidx in edge_store.items():
            data[et].edge_index = torch.tensor(eidx, dtype=torch.long)
            data[et].edge_attr = torch.tensor(
                edge_attr_store[et],
                dtype=torch.float32
            )
            
        label_dict = getattr(self, "labels_cache", None)

        if label_dict is not None and idx in label_dict:
            for t in ['satellite', 'hap', 'ground']:
                if t in data.node_types:
                    data[t].y = label_dict[idx][t]

        self.cache[idx] = data
        return data
    
    def precompute_labels(self):
        print("[Dataset] Precomputing labels...")
        
        self.labels_cache = {}
        for idx in range(self.num_samples):            
            print(f"[Dataset] processing sample {idx}/{self.num_samples}")
            
            p = self.lhs[idx]
            clients_n = 125

            nodes = generate_nodes(
                num_sats=9, num_uavs=2, num_ground=5, 
                num_clients=clients_n, 
                area_size=self.area_size, 
                gradient_dim=self.gradient_dim
            )
            
            clients = [n for n in nodes if n.type == NodeType.CLIENT]
            candidates = [n for n in nodes if n.type != NodeType.CLIENT]
            
            selected = []
            for tier in [NodeType.GROUND, NodeType.UAV, NodeType.SATELLITE]:
                tier_nodes = [n for n in candidates if n.type == tier]
                if tier_nodes:
                    selected.append(tier_nodes[self.rng.integers(len(tier_nodes))])

            base_utility = compute_utility(
                selected, clients, alpha=0.65, beta=0.35, delta_list=[0.1, 0.05]
            )

            # Label generation per tier
            placement_types = ['satellite', 'hap', 'ground']
            type_map = {
                    NodeType.SATELLITE: 'satellite',
                    NodeType.UAV: 'hap',
                    NodeType.GROUND: 'ground'
                }
            
            label_dict = {}
            for t in placement_types:
                tier_nodes = [n for n in candidates if type_map.get(n.type) == t]
                labels = []
                for candidate in tier_nodes:
                    new_servers = list(selected)
                    if candidate not in new_servers:
                        new_servers.append(candidate)
                    new_utility = compute_utility(new_servers, clients, alpha=0.65, beta=0.35, delta_list=[0.1, 0.05])
                    marginal_gain = base_utility - new_utility
                    labels.append(float(marginal_gain))

                labels = np.asarray(labels, dtype=np.float32)
                if len(labels) > 1:
                    labels = (labels - labels.mean()) / (labels.std() + 1e-6)
                    
                label_dict[t] = torch.tensor(labels, dtype=torch.float32)
                
            self.labels_cache[idx] = label_dict

            if idx % 50 == 0:
                print(f"[Dataset] Precomputed {idx}/{self.num_samples}")

        print("[Dataset] Done precomputing labels.")


def get_gnn_datasets(num_samples=14000, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Returns train, val, test datasets with proper split ratios."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    full_dataset = SAGINSnapshotDataset(num_samples=num_samples, lhs_seed=seed)

    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size

    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size],
                                                generator=torch.Generator().manual_seed(seed))

    # Attach the same labels cache to all splits
    full_dataset.precompute_labels()
    for ds in (train_ds, val_ds, test_ds):
        ds.dataset.labels_cache = full_dataset.labels_cache  # share labels

    return train_ds, val_ds, test_ds
