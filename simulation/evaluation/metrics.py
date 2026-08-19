from __future__ import annotations
import logging

import numpy as np

log = logging.getLogger(__name__)


def compute_e2e_latency(clients, servers, t_now=None):
    vals=[]
    for c in clients:
        if not servers:
            continue
        # Use min latency to selected servers to reflect actual connection latency
        server_lats = [c.get_latency_to(s, t_now) for s in servers]
        vals.append(float(min(server_lats)))
    
    # `get_latency_to` already returns one-way latency in milliseconds.
    # Return the mean of median per-client latencies.
    return float(np.mean(vals)) if vals else 0.0


def compute_total_energy(clients, selected_servers, ota_params, transmission_time=1.0, t_now=None):
    e = 0.0
    max_power_clients = 0
    power_stats = []

    for c in clients:
        best_server = None
        best_snr = -float("inf")

        for s in selected_servers:
            try:
                snr = c.compute_snr_to(s, t_now)
            except Exception:
                snr = 0.0
            if snr > best_snr:
                best_snr = snr
                best_server = s

        if best_server is not None:
            cmap = ota_params.get(best_server.id, {})
            tx_power = float(cmap.get(c.id, {}).get("power", c.power))
        else:
            tx_power = c.power

        power_stats.append(tx_power)
        e += tx_power * transmission_time

        if tx_power >= c.power - 1e-6:
            max_power_clients += 1

    # Server infrastructure energy (dominant in SAGIN: satellites at 500W, BSs at 40W)
    server_energy = sum(getattr(s, "power", 0.0) * transmission_time for s in selected_servers)
    e += server_energy

    if clients:
        avg_power_ratio = np.mean([p / c.power for p, c in zip(power_stats, clients)])
        log.info(
            f"Energy: {e:.4f} J (client={e - server_energy:.1f}J + server={server_energy:.1f}J) "
            f"| Max-power clients: {max_power_clients}/{len(clients)} "
            f"| Avg power ratio: {avg_power_ratio:.3f}"
        )

    return e


def compute_cvar(loss_history, alpha=0.95):
    """
    Compute CVaR (Conditional Value-at-Risk) per Eq. 15.

    CVaR_α(L) = min_t { t + 1/(1-α) E[(L-t)⁺] }

    For empirical data with confidence level α (e.g., 0.95 for worst 5% tail):
    Sort ascending, take the upper (1-α) tail.

    Args:
        loss_history: Array of loss values
        alpha: Confidence level (0.95 for worst 5% tail per paper Eq. 15)
    """
    if len(loss_history) == 0:
        return 0.0

    arr = np.sort(np.asarray(loss_history, dtype=float))
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0

    # For worst (1-α) fraction (upper tail)
    k = max(1, int(np.ceil((1 - alpha) * len(arr))))
    tail = arr[-k:]
    return float(np.mean(tail))
