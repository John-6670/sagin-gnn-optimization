from __future__ import annotations

from pathlib import Path
import torch
import yaml
from torch_geometric.loader import DataLoader

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

    # Load from config if not manually specified
    if ckpt_path is None:
        cfg_path = Path("configs/default.yaml")

        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            ckpt_path = (cfg.get("gnn") or {}).get("checkpoint")

    # No checkpoint available
    if not ckpt_path:
        print("[WARN] No GNN checkpoint specified")
        return None

    path = Path(ckpt_path)

    if not path.exists():
        print(f"[WARN] Checkpoint not found: {path}")
        return None

    # Build temporary dataset sample to infer metadata
    ds = SAGINSnapshotDataset(num_samples=1)

    if len(ds) == 0:
        print("[WARN] Empty SAGIN dataset")
        return None

    sample = ds[0]

    # Infer feature dimensions
    in_dims = {}

    for node_type in sample.node_types:
        in_dims[node_type] = sample[node_type].x.shape[1]

    # Create model
    model = SAGINHeteroGNN(
        metadata=sample.metadata(),
        in_dims=in_dims,
    )

    # Load weights
    state = torch.load(path, map_location="cpu")

    # Support both raw state_dict and wrapped checkpoints
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)

    model.eval()

    print(f"[INFO] Loaded HGNN checkpoint from {path}")

    return model


def predict_candidate_scores(model, candidates, clients, selected_servers=None):
    if model is None:
        return {c: 1.0 for c in candidates}

    if selected_servers is None:
        selected_servers = []

    graph_data, node_maps = build_graph_snapshot(
        candidates=candidates,
        clients=clients,
        selected_servers=selected_servers,
    )

    placement_mask = torch.zeros(
        sum(graph_data[k].x.size(0) for k in graph_data.node_types),
        dtype=torch.bool,
    )

    model.eval()

    with torch.no_grad():
        preds = model(graph_data, placement_mask)

    scores = {}

    for node_type, mapping in node_maps.items():
        if node_type not in preds:
            continue

        pred_tensor = preds[node_type]

        for idx, node in mapping.items():
            scores[node] = float(pred_tensor[idx].item())

    return scores


if __name__ == '__main__':
    train_gnn()
