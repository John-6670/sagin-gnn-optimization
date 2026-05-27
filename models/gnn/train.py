from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

from models.gnn.dataset import SAGINSnapshotDataset
from models.gnn.hgnn import SAGINHeteroGNN, SubmodularityRegularizer, SNRGradientPenalty


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
            loss.backward()
            opt.step()
        sch.step()

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
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
    ckpt_path = checkpoint
    if ckpt_path is None:
        cfg_path = Path('configs/default.yaml')
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            ckpt_path = (cfg.get('gnn') or {}).get('checkpoint')

    if not ckpt_path:
        return None

    path = Path(ckpt_path)
    if not path.exists():
        return None

    _ = torch.load(path, map_location='cpu')
    return {"checkpoint": str(path)}


def predict_candidate_scores(model, candidates, clients):
    if not candidates:
        return {}

    if not clients:
        return {c: 1.0 for c in candidates}

    avg_snrs: Dict = {}
    avg_lats: Dict = {}
    tier_bonus = {
        'ground': 0.02,
        'uav': 0.05,
        'sat': 0.03,
    }

    for c in candidates:
        snrs: List[float] = []
        lats: List[float] = []
        for k in clients:
            snrs.append(float(max(k.compute_snr_to(c), 1e-12)))
            lats.append(float(max(k.get_latency_to(c), 1e-9)))
        avg_snrs[c] = float(np.mean(snrs))
        avg_lats[c] = float(np.mean(lats))

    snr_vals = np.array(list(avg_snrs.values()), dtype=float)
    lat_vals = np.array(list(avg_lats.values()), dtype=float)

    snr_norm = (snr_vals - snr_vals.min()) / (snr_vals.ptp() + 1e-12)
    lat_norm = (lat_vals - lat_vals.min()) / (lat_vals.ptp() + 1e-12)

    scores = {}
    for idx, c in enumerate(candidates):
        score = 0.7 * snr_norm[idx] + 0.3 * (1.0 - lat_norm[idx])
        score += tier_bonus.get(c.type.value, 0.0)
        scores[c] = float(score)
    return scores


if __name__ == '__main__':
    train_gnn()
