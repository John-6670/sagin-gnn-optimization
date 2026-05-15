from __future__ import annotations

import numpy as np

from simulation.topology.aircomp import compute_amse_n


def identify_stragglers(clients, server, snr_map):
    snrs = np.array([snr_map[c][server] for c in clients], dtype=float)
    q25, q75 = np.percentile(snrs, [25, 75])
    iqr = q75 - q25
    tau = q25 - 1.5 * iqr
    stragglers = [c for c in clients if snr_map[c][server] < tau]
    majority = [c for c in clients if c not in stragglers]
    return stragglers, majority, float(tau)


def aircomp_aggregate(clients, server, snr_map, delta_list):
    stragglers, majority, _ = identify_stragglers(clients, server, snr_map)
    if not majority:
        majority = clients

    weighted = []
    weights = []
    snr_dict = {}
    for c in majority:
        h = c.get_channel(server)
        p = c.power
        g = np.random.randn(c.gradient_dim)
        w = np.abs(h) * np.sqrt(max(p, 1e-12))
        weighted.append(w * g)
        weights.append(w)
        snr_dict[c] = max(snr_map[c][server], 1e-12)

    denom = np.sum(weights) + 1e-12
    g_majority = np.sum(weighted, axis=0) / denom
    amse_analog = compute_amse_n(snr_dict, sigma2=np.mean([c.noise_variance for c in majority]), d=majority[0].gradient_dim)
    return g_majority, amse_analog, stragglers, majority


def digital_compress(gradient, num_bits=8):
    g = np.asarray(gradient)
    gmin, gmax = g.min(), g.max()
    if np.isclose(gmax, gmin):
        return g.copy(), 0.0
    delta = (gmax - gmin) / (2**num_bits - 1)
    q = np.round(g / delta) * delta
    eps_q = (delta**2) / 12.0
    return q, float(eps_q)


def hybrid_patch(clients, server, snr_map, delta_list, num_bits=8):
    g_majority, amse_analog, stragglers, majority = aircomp_aggregate(clients, server, snr_map, delta_list)
    K = len(clients)
    if K == 0:
        return np.array([]), 0.0, {"stragglers": [], "majority": []}

    if stragglers:
        compressed = []
        eps = []
        for c in stragglers:
            gk = np.random.randn(c.gradient_dim)
            q, eq = digital_compress(gk, num_bits=num_bits)
            compressed.append(q)
            eps.append(eq)
        g_digital = np.mean(compressed, axis=0)
        eps_q = float(np.mean(eps))
    else:
        g_digital = np.zeros_like(g_majority)
        eps_q = 0.0

    w_a = len(majority) / K
    w_d = len(stragglers) / K
    g_hybrid = w_a * g_majority + w_d * g_digital
    amse_bound = (w_a**2) * amse_analog + (w_d**2) * eps_q
    return g_hybrid, float(amse_bound), {"stragglers": stragglers, "majority": majority, "w_a": w_a, "w_d": w_d}
