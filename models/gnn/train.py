from __future__ import annotations

from pathlib import Path
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
    return None


def predict_candidate_scores(model, candidates, clients):
    return {c: 0.0 for c in candidates}


if __name__ == '__main__':
    train_gnn()
