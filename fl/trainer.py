from __future__ import annotations
import copy, math
import numpy as np
import torch
from simulation.topology.patching import hybrid_patch, aircomp_aggregate

from .device import DEVICE


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


def FederatedRound(
    round_idx, clients, servers, ota_params, task, global_model,
    client_loaders, test_loader, snr_map, delta_list, use_hybrid=True
):
    p_t = 0.1 + 0.3 * math.sin(2*math.pi*round_idx/24.0)
    p_t = float(np.clip(p_t, 0.01, 0.95))
    active = [i for i in range(len(clients)) if np.random.rand() < p_t]
    if not active:
        active = [np.random.randint(0, len(clients))]

    grads=[]
    for cid in active:
        local = copy.deepcopy(global_model).to(DEVICE)
        if task.optimizer == 'sgd':
            opt = torch.optim.SGD(local.parameters(), lr=task.lr, **task.optimizer_kwargs)
        elif task.optimizer == 'adam':
            opt = torch.optim.Adam(local.parameters(), lr=task.lr, **task.optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer {task.optimizer}")
        
        local.train()
        for _ in range(task.local_epochs):
            for xb, yb in client_loaders[cid]:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                
                opt.zero_grad()
                out = local(xb)
                loss = task.criterion(out,yb)
                loss.backward()
                
                if task.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(local.parameters(), task.grad_clip)
                opt.step()
        
        diff=[]
        for pg, pl in zip(global_model.parameters(), local.parameters()):
            diff.append((pg.data - pl.data).flatten())
        grads.append(torch.cat(diff).detach().cpu().numpy())

    if use_hybrid:
        _, amse_round, meta = hybrid_patch([clients[i] for i in active], servers[0], snr_map, delta_list)
        w = np.array([meta['w_a'], meta['w_d']])
        g = np.mean(grads, axis=0)*w.sum()
    else:
        g, amse_round, _, _ = aircomp_aggregate([clients[i] for i in active], servers[0], snr_map, delta_list)

    ptr = 0
    for p in global_model.parameters():
        n = p.numel()
        upd = torch.tensor(g[ptr:ptr+n], dtype=p.dtype, device=DEVICE).view_as(p)
        p.data -= task.lr * upd
        ptr += n

    loss, acc = _eval(global_model, test_loader, task.criterion)
    return {
        'round': round_idx,
        'active_clients': len(active),
        'accuracy': acc,
        'loss': loss,
        'amse': amse_round,
        'p_t': p_t
    }
