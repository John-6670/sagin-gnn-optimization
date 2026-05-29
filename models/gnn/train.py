from __future__ import annotations

from pathlib import Path
import torch
import yaml
from torch_geometric.loader import DataLoader
import pickle

from models.gnn.dataset import SAGINSnapshotDataset
from models.gnn.hgnn import SAGINHeteroGNN, SubmodularityRegularizer, SNRGradientPenalty
from graph.graph_builder import build_graph_snapshot


def train_gnn(epochs=10, batch_size=8, lr=3e-4, checkpoint_path='checkpoints/sagin_hgnn.pt'):
    ds = SAGINSnapshotDataset(num_samples=200)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    sample = ds[0]
    in_dims = {k: sample[k].x.shape[1] for k in sample.node_types}
    model = SAGINHeteroGNN(sample.metadata(), in_dims=in_dims)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    mse = torch.nn.MSELoss()
    subreg = SubmodularityRegularizer()
    snrreg = SNRGradientPenalty()

    model.train()
    for _ in range(epochs):
        for batch in loader:
            opt.zero_grad()
            placement_mask = torch.zeros(sum(batch[k].x.size(0) for k in batch.node_types), dtype=torch.bool)
            preds = model(batch, placement_mask)
            loss = 0.0
            for nt in preds:
                y = torch.zeros_like(preds[nt])
                loss = loss + mse(preds[nt], y)
                loss = loss + 0.1 * subreg(preds[nt], preds[nt].roll(1, 0))
                loss = loss + 0.05 * snrreg(preds[nt], batch[nt].x)
                
            for nt in preds:
                if hasattr(batch[nt], 'y'):
                    y = batch[nt].y.float().to(preds[nt].device)
                else:
                    y = torch.zeros_like(preds[nt])

                loss = loss + mse(preds[nt], y)

                shifted = preds[nt].roll(1, 0)
                loss = loss + 0.1 * subreg(preds[nt], shifted)

                x = batch[nt].x.detach().clone().requires_grad_(True)
                loss = loss + 0.05 * snrreg(preds[nt], x)
            loss.backward()
            opt.step()
        sch.step()

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    meta_path = Path(checkpoint_path).with_suffix('.meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump({'metadata': sample.metadata(), 'in_dims': in_dims}, f)
    _write_checkpoint_to_config(checkpoint_path)
    return checkpoint_path


def _write_checkpoint_to_config(path):
    cfg_path = Path('configs/default.yaml')
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.setdefault('gnn', {})
    cfg['gnn']['checkpoint'] = path
    cfg['gnn'].setdefault('kappa', 0.15)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def load_gnn_for_inference(checkpoint=None):
    if checkpoint is None:
        checkpoint = 'checkpoints/sagin_hgnn.pt'
    ckpt_path = Path(checkpoint)
    meta_path = ckpt_path.with_suffix('.meta.pkl')
    if not ckpt_path.exists() or not meta_path.exists():
        return None
    try:
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        m = SAGINHeteroGNN(meta['metadata'], in_dims=meta['in_dims'])
        m.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        m.eval()
        return m
    except Exception as e:
        print(f"[GNN] load failed: {e}")
        return None


def predict_candidate_scores(model, candidates, clients, selected_servers=None):
    if model is None:
        return {c: 0.0 for c in candidates}
    import numpy as np
    from torch_geometric.data import HeteroData
    from simulation.topology.nodes import NodeType

    type_map = {NodeType.SATELLITE: 'satellite', NodeType.UAV: 'hap',
                NodeType.GROUND: 'ground', NodeType.CLIENT: 'client'}
    data = HeteroData()
    node_to_local = {}
    for t in type_map.values():
        data[t].x = []

    for n in list(candidates) + list(clients):
        tname = type_map[n.type]
        feat = np.array([*n.position, n.power, n.noise_variance,
                         float(n.type == NodeType.CLIENT)], dtype=np.float32)
        node_to_local[n.id] = (tname, len(data[tname].x))
        data[tname].x.append(feat)

    for t in type_map.values():
        rows = data[t].x
        data[t].x = (torch.tensor(np.array(rows), dtype=torch.float32)
                     if rows else torch.zeros((0, 6), dtype=torch.float32))

    edge_store = {}
    for c in clients:
        for s in candidates:
            if c.compute_snr_to(s) < 1e-6:
                continue
            st, si = node_to_local[c.id]
            dt, di = node_to_local[s.id]
            et = (st, 'link', dt)
            edge_store.setdefault(et, [[], []])
            edge_store[et][0].append(si)
            edge_store[et][1].append(di)
    for et, eidx in edge_store.items():
        data[et].edge_index = torch.tensor(eidx, dtype=torch.long)

    placement_mask = torch.zeros(
        sum(data[k].x.size(0) for k in data.node_types), dtype=torch.bool)
    try:
        with torch.no_grad():
            preds = model(data, placement_mask)
    except Exception as e:
        print(f"[GNN] inference error: {e}")
        return {c: 0.0 for c in candidates}

    scores = {}
    for c in candidates:
        tname, idx = node_to_local[c.id]
        scores[c] = float(preds[tname][idx].item()) if (
            tname in preds and idx < preds[tname].size(0)) else 0.0
    return scores


if __name__ == '__main__':
    train_gnn()
