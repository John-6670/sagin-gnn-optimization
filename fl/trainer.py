from __future__ import annotations
import copy
import logging
import numpy as np
import torch
from simulation.topology.patching import hybrid_patch, aircomp_aggregate

from fl.convergence import ConvergenceTracker
from .device import DEVICE

log = logging.getLogger("fl.trainer")


def _eval(model, test_loader, criterion):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            out = model(xb)
            l = criterion(out, yb)
            loss += l.item() * len(yb)
            total += len(yb)
            
            # Only compute accuracy for classification (targets are long integers)
            if yb.dtype == torch.long and out.ndim == 2:
                correct += (out.argmax(1) == yb).sum().item()
    
    acc = correct / max(total, 1)
    return loss / max(total, 1), acc

CONVERGENCE_TRACKER = ConvergenceTracker()

def FederatedRound(
    round_idx, clients, servers, ota_params, task, global_model,
    client_loaders, test_loader, snr_map, delta_list, use_hybrid=True,
    t_now=None
):
    p_t = 1.0
    active = list(range(len(clients)))

    log.debug("  Round %d: all clients active (%d/%d)",
              round_idx, len(active), len(clients))

    grads = []
    for cid in active:
        local = copy.deepcopy(global_model).to(DEVICE)
        if task.optimizer == 'sgd':
            opt = torch.optim.SGD(local.parameters(), lr=task.lr, **task.optimizer_kwargs)
        elif task.optimizer == 'adam':
            opt = torch.optim.Adam(local.parameters(), lr=task.lr, **task.optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer {task.optimizer}")

        local.train()
        epoch_losses = []
        for epoch in range(task.local_epochs):
            batch_loss = 0.0
            n_batches = 0
            for xb, yb in client_loaders[cid]:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                out = local(xb)
                loss = task.criterion(out, yb)
                loss.backward()
                if task.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(local.parameters(), task.grad_clip)
                opt.step()
                batch_loss += loss.item()
                n_batches += 1
            epoch_losses.append(batch_loss / max(n_batches, 1))
        log.debug("    client %d: epoch losses %s", cid, [f"{l:.4f}" for l in epoch_losses])

        diff = []
        for pg, pl in zip(global_model.parameters(), local.parameters()):
            diff.append((pg.data - pl.data).flatten())
        grad_vec = torch.cat(diff).detach().cpu().numpy()
        log.debug("    client %d: grad norm=%.6f", cid, float(np.linalg.norm(grad_vec)))
        grads.append(grad_vec)

    log.debug("  Round %d: aggregating %d gradients (use_hybrid=%s)...", round_idx, len(grads), use_hybrid)

    # Extract phase corrections from ota_params if available
    phase_corrections = {}
    if ota_params and isinstance(ota_params, dict):
        # ota_params structure: {server_id: {client_id: {"phase": ...}}}
        for server_params in ota_params.values():
            if isinstance(server_params, dict):
                for client_id, params in server_params.items():
                    if isinstance(params, dict) and "phase" in params:
                        phase_corrections[client_id] = params["phase"]

    if use_hybrid:
        _, amse_round, meta = hybrid_patch(
            [clients[i] for i in active], servers[0], snr_map, delta_list,
            phase_corrections=phase_corrections, t_now=t_now
        )
        w = np.array([meta['w_a'], meta['w_d']])
        base_gradient = np.mean(grads, axis=0)

        noise_std = np.sqrt(max(amse_round, 1e-12))
        aggregation_noise = np.random.normal(
            0.0,
            noise_std,
            size=base_gradient.shape,
        )

        g = (base_gradient + aggregation_noise) * w.sum()
        log.debug("  Round %d: hybrid_patch amse=%.6f  w_a=%.4f  w_d=%.4f  w_sum=%.4f",
                  round_idx, amse_round, meta['w_a'], meta['w_d'], w.sum())
    else:
        g, amse_round, _, _ = aircomp_aggregate(
            [clients[i] for i in active], servers[0], snr_map, delta_list,
            phase_corrections=phase_corrections, t_now=t_now
        )
        log.debug("  Round %d: aircomp amse=%.6f", round_idx, amse_round)

    log.debug("  Round %d: applying global model update (grad norm=%.6f)...",
              round_idx, float(np.linalg.norm(g)))
    ptr = 0
    for p in global_model.parameters():
        n = p.numel()
        upd = torch.tensor(g[ptr:ptr+n], dtype=p.dtype, device=DEVICE).view_as(p)
        p.data -= task.server_lr * upd
        ptr += n

    loss, acc = _eval(global_model, test_loader, task.criterion)
    log.debug("  Round %d: eval complete — loss=%.6f  acc=%.4f", round_idx, loss, acc)
    CONVERGENCE_TRACKER.update(
        round_idx,
        loss,
        acc,
        amse_round,
    )
    return {
        'round': round_idx,
        'active_clients': len(active),
        'accuracy': acc,
        'loss': loss,
        'amse': amse_round,
        'p_t': p_t
    }
