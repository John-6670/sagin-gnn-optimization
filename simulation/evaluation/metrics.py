from __future__ import annotations
import numpy as np


def compute_e2e_latency(clients, servers, t_now=None):
    vals=[]
    for c in clients:
        if not servers:
            continue
        vals.append(min(c.get_latency_to(s, t_now) for s in servers))
    
    return float(np.mean(vals)) if vals else 0.0


def compute_total_energy(clients, ota_params, transmission_time=1.0):
    e = 0.0
    for s_id, cmap in ota_params.items():
        for c in clients:
            if c.id in cmap:
                e += float(cmap[c.id].get("power", 0.0)) * transmission_time
    return e


def compute_cvar(loss_history, alpha=0.05):
    if len(loss_history) == 0:
        return 0.0
    
    arr = np.sort(np.asarray(loss_history, dtype=float))[::-1]
    k = max(1,int(np.ceil(alpha*len(arr))))
    return float(np.mean(arr[:k]))
