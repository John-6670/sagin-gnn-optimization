from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
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
        p = self.lhs[idx]
        clients_n = int(100 + p[0] * 400)
        sats = 36
        uavs = 20
        ground = 40
        t0 = None
        nodes = generate_nodes(sats, uavs, ground, clients_n, self.area_size, self.gradient_dim, t0=t0)
        clients = [n for n in nodes if n.type == NodeType.CLIENT]
        candidates = [n for n in nodes if n.type != NodeType.CLIENT]

        data = HeteroData()
        type_map = {NodeType.SATELLITE: 'satellite', NodeType.UAV: 'hap', NodeType.GROUND: 'ground', NodeType.CLIENT: 'client'}
        node_to_local = {}
        for t in type_map.values():
            data[t].x = []

        for n in nodes:
            tname = type_map[n.type]
            feat = np.array([*n.position, n.power, n.noise_variance, float(n.type == NodeType.CLIENT)], dtype=np.float32)
            node_to_local[n.id] = (tname, len(data[tname].x))
            data[tname].x.append(feat)

        for t in type_map.values():
            data[t].x = torch.tensor(np.array(data[t].x), dtype=torch.float32)

        edge_store = {}
        for c in clients:
            for s in candidates:
                snr = c.compute_snr_to(s)
                if snr < self.snr_threshold:
                    continue
                src_t, src_i = node_to_local[c.id]
                dst_t, dst_i = node_to_local[s.id]
                et = (src_t, 'link', dst_t)
                edge_store.setdefault(et, [[], []])
                edge_store[et][0].append(src_i)
                edge_store[et][1].append(dst_i)

        for et, eidx in edge_store.items():
            data[et].edge_index = torch.tensor(eidx, dtype=torch.long)

        S = candidates[: max(1, len(candidates)//10)]
        base = compute_utility(S, clients, 0.5, 0.5, [0.1, 0.2])
        labels = {}
        for s in candidates:
            gain = base - compute_utility(S + [s], clients, 0.5, 0.5, [0.1, 0.2])
            labels[s.id] = gain
        data['labels'] = labels
        return data
