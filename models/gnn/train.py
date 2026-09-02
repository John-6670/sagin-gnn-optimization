from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader
import pickle
from torch.utils.data import Dataset

from models.gnn.hgnn import SAGINHeteroGNN, SubmodularityRegularizer, SNRGradientPenalty
from simulation.config_loader import load_config

# Load config once at module level
_config = load_config()


class PrecomputedGraphDataset(Dataset):
    def __init__(self, path="precomputed_dataset", split="train"):
        self.files = sorted(Path(path).glob(f"sample_{split}_*.pt"))
        if not self.files:
            self.files = sorted(Path(path).glob("sample_*.pt"))  # fallback
        print(f"[Dataset] Loaded {len(self.files)} {split} samples from {path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])


def train_gnn(epochs=None, batch_size=None, lr=None, checkpoint_path=None):
    # Load GNN training parameters from config if not provided
    gnn_cfg = _config.get('gnn', {})
    epochs = epochs if epochs is not None else gnn_cfg.get('epochs', 200)
    batch_size = batch_size if batch_size is not None else gnn_cfg.get('batch_size', 32)
    lr = lr if lr is not None else gnn_cfg.get('lr', 1e-3)
    checkpoint_path = checkpoint_path if checkpoint_path is not None else gnn_cfg.get('checkpoint', 'checkpoints/sagin_hgnn.pt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GNN Training] Using device: {device} | CUDA available: {torch.cuda.is_available()}")

    # === Load Precomputed Datasets (No generation here) ===
    train_ds = PrecomputedGraphDataset(path="precomputed_dataset", split="train")
    val_ds   = PrecomputedGraphDataset(path="precomputed_dataset", split="val")
    test_ds  = PrecomputedGraphDataset(path="precomputed_dataset", split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              pin_memory=True, num_workers=2)

    # Load one sample for metadata
    sample = torch.load("precomputed_dataset/sample_train_0.pt") if train_ds.files else torch.load("precomputed_dataset/sample_0.pt")
    in_dims = {k: sample[k].x.shape[1] for k in sample.node_types}

    model = SAGINHeteroGNN(sample.metadata(), in_dims=in_dims).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    mse = torch.nn.MSELoss()
    submod_reg = SubmodularityRegularizer()
    snr_penalty = SNRGradientPenalty()
    best_val_loss = float('inf')
    no_improve_epochs = 0
    patience = 20
    accumulation_steps = 2

    print("=" * 70)
    print(f"Dataset → Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Batch size: {batch_size} | Epochs: {epochs}")
    print("=" * 70)

    for epoch in range(epochs):
        # ====================== TRAINING ======================
        model.train()
        epoch_loss = 0.0

        opt.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            for nt in batch.node_types:
                if hasattr(batch[nt], 'x'):
                    batch[nt].x.requires_grad_(True)

            total_nodes = sum(batch[k].x.size(0) for k in batch.node_types)
            placement_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)

            preds = model(batch, placement_mask)

            loss = 0.0
            for nt in preds:
                if not hasattr(batch[nt], 'y') or batch[nt].y is None:
                    continue
                y = batch[nt].y.float().view(-1)
                pred = preds[nt].view(-1)
                min_len = min(pred.size(0), y.size(0))

                if min_len == 0:
                    continue

                mse_part = mse(pred[:min_len], y[:min_len])
                lsub_part = torch.tensor(0.0, device=device)
                lsnr_part = torch.tensor(0.0, device=device)

                # Submodularity regularization: sample A⊂B pairs
                # For now, use sequential approximation as proxy (batch order)
                if min_len > 1:
                    lsub_part = submod_reg(pred[:min_len - 1], pred[1:min_len])

                # SNR gradient penalty: use actual SNR feature (index 4: mean_snr after P0.3)
                # Before P0.3: index 2 is noise_variance. After: SNR stats at indices 4-7
                snr_feature_idx = 2  # Current noise_variance feature
                if batch[nt].x.size(1) > snr_feature_idx:
                    snr_feature = batch[nt].x[:min_len, snr_feature_idx]
                    if snr_feature.requires_grad:
                        lsnr_part = snr_penalty(pred[:min_len], snr_feature)

                loss += mse_part + gnn_cfg.get('loss_weight_submod', 0.1) * lsub_part + gnn_cfg.get('loss_weight_snr_grad', 0.05) * lsnr_part

            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            else:
                grad_norm = float('nan')

            epoch_loss += loss.item() * accumulation_steps

            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch+1:02d}] Batch {batch_idx:03d}/{len(train_loader)} | "
                      f"Loss={loss.item() * accumulation_steps:.6f} | Grad={grad_norm:.4f}")

        # ====================== VALIDATION ======================
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                placement_mask = torch.zeros(
                    sum(batch[k].x.size(0) for k in batch.node_types),
                    dtype=torch.bool, device=device
                )
                preds = model(batch, placement_mask)
                batch_loss = 0.0
                for nt in preds:
                    if not hasattr(batch[nt], 'y') or batch[nt].y is None:
                        continue
                    y = batch[nt].y.float().view(-1)
                    pred = preds[nt].view(-1)
                    min_len = min(pred.size(0), y.size(0))
                    batch_loss += mse(pred[:min_len], y[:min_len]).item()
                val_loss += batch_loss

        val_loss /= max(len(val_loader), 1)
        avg_train_loss = epoch_loss / len(train_loader)

        # ====================== SCHEDULER + LOGGING ======================
        sch.step()
        current_lr = sch.get_last_lr()[0]

        print(
            f"\n[Epoch {epoch+1:02d}/{epochs}] "
            f"AvgTrainLoss={avg_train_loss:.6f} | "
            f"ValLoss={val_loss:.6f} | "
            f"LR={current_lr:.8f}"
        )

        if torch.cuda.is_available():
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  → Best model saved (ValLoss = {val_loss:.6f})")
        else:
            no_improve_epochs += 1
            print(f"  → No improvement for {no_improve_epochs}/{patience} epochs")
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # ====================== FINAL SAVE ======================
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    meta_path = Path(checkpoint_path).with_suffix('.meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump({'metadata': sample.metadata(), 'in_dims': in_dims}, f)

    _write_checkpoint_to_config(checkpoint_path)
    print(f"\n[GNN] Training completed. Best model saved to {checkpoint_path}")
    print(f"Final Best Validation Loss: {best_val_loss:.6f}")
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
    from optimization.objective import compute_orbital_avg_snr

    type_map = {NodeType.SATELLITE: 'satellite', NodeType.UAV: 'hap',
                NodeType.GROUND: 'ground', NodeType.CLIENT: 'client'}
    data = HeteroData()
    node_to_local = {}
    for t in type_map.values():
        data[t].x = []

    # Pre-compute SNR statistics for server nodes
    for n in list(candidates) + list(clients):
        tname = type_map[n.type]

        if n.type != NodeType.CLIENT:
            # Server nodes: compute SNR statistics to all clients
            # Use orbital-average SNR for satellite links per Eq. 21
            snr_values = []
            for c in clients:
                if n.type == NodeType.SATELLITE:
                    snr = compute_orbital_avg_snr(c, n, t0=None)  # Uses current time
                else:
                    snr = c.compute_snr_to(n)
                if np.isfinite(snr) and snr > 0:
                    snr_values.append(snr)

            if snr_values:
                snr_mean = float(np.mean(snr_values))
                snr_max = float(np.max(snr_values))
                snr_min = float(np.min(snr_values))
                snr_std = float(np.std(snr_values))
            else:
                snr_mean = snr_max = snr_min = snr_std = 0.0
        else:
            # Client nodes: SNR stats not applicable
            snr_mean = snr_max = snr_min = snr_std = 0.0

        # Paper Eq. 31: x_v^t = [pos_v^t, vel_v^t, SNR_v^t, load_v^t, tier_v]
        velocity = n.get_velocity_at()
        feat = np.array([
            *n.position,                    # 3: position
            *velocity,                       # 3: velocity
            snr_mean, snr_max, snr_min, snr_std,  # 4: SNR statistics
            n.load,                          # 1: load
            float(n.type == NodeType.SATELLITE),
            float(n.type == NodeType.UAV),
            float(n.type == NodeType.GROUND),
            float(n.type == NodeType.CLIENT),
        ], dtype=np.float32)

        node_to_local[n.id] = (tname, len(data[tname].x))
        data[tname].x.append(feat)

    for t in type_map.values():
        rows = data[t].x
        data[t].x = (torch.tensor(np.array(rows), dtype=torch.float32)
                     if rows else torch.zeros((0, 15), dtype=torch.float32))

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

    for c in candidates:
        if c.type == NodeType.SATELLITE:
            scores[c] *= 0.6      # downweight sats
        elif c.type == NodeType.UAV:
            scores[c] *= 0.85

    return scores


if __name__ == '__main__':
    train_gnn()
