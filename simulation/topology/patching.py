from __future__ import annotations

import numpy as np

from simulation.topology.aircomp import compute_amse_n


def identify_stragglers(clients, server, snr_map):
    """Identify stragglers using IQR-based detection per Algorithm 4.

    Operates on log10(SNR) scale (dB) since SNR values span many orders
    of magnitude, making linear-scale IQR ineffective.

    Args:
        clients: List of client nodes
        server: Server node
        snr_map: Dict mapping client -> {server: SNR}

    Returns:
        tuple: (stragglers, majority, threshold_in_linear_scale)
    """
    snrs = np.array([max(snr_map[c][server], 1e-12) for c in clients], dtype=float)

    # Compute IQR on log10(SNR) scale (dB equivalent)
    log_snrs = np.log10(snrs)
    q25, q75 = np.percentile(log_snrs, [25, 75])
    iqr = q75 - q25
    tau_log = q25 - 1.5 * iqr
    tau = 10.0 ** tau_log  # Convert back to linear scale

    stragglers = [c for c in clients if snr_map[c][server] < tau]
    majority = [c for c in clients if c not in stragglers]
    return stragglers, majority, float(tau)


def aircomp_aggregate(clients, server, snr_map, delta_list, phase_corrections=None, t_now=None):
    """
    Analog aggregation with coherent combining using pre-equalization phases.

    Per Algorithm 3 Step 3: Apply pre-equalization phase θ_k = -arg(α_k) - 2π f_D τ_k
    so that all signals coherently combine at the aggregator.

    Args:
        clients: List of client nodes
        server: Server node
        snr_map: Dict mapping client -> {server: SNR}
        delta_list: List of cascaded errors per tier
        phase_corrections: Dict mapping client_id -> pre_equalization_phase (radians)
        t_now: Current time (skyfield Time object)

    Returns:
        tuple: (g_majority, amse_analog, stragglers, majority)
    """
    stragglers, majority, _ = identify_stragglers(clients, server, snr_map)
    if not majority:
        majority = clients

    weighted = []
    weights = []
    snr_dict = {}
    for c in majority:
        h = c.get_channel(server, t_now)
        p = c.power
        g = np.random.randn(c.gradient_dim)

        # Apply pre-equalization phase for coherent combining
        # w = |h|√p * exp(j*phase_correction)
        w_mag = np.abs(h) * np.sqrt(max(p, 1e-12))
        if phase_corrections is not None and c.id in phase_corrections:
            phase = phase_corrections[c.id]
            w = w_mag * np.exp(1j * phase)
        else:
            w = w_mag
        weighted.append(w * g)
        weights.append(w)
        # Use the SNR without phase for AMSE computation
        snr_dict[c] = max(snr_map[c][server], 1e-12)

    denom = np.sum(np.abs(weights)) + 1e-12
    g_majority = np.sum(weighted, axis=0) / denom
    amse_analog = compute_amse_n(snr_dict, sigma2=np.mean([c.noise_variance for c in majority]), d=majority[0].gradient_dim)
    return g_majority, amse_analog, stragglers, majority


def digital_compress(gradient, num_bits=8):
    """Quantize gradient to num_bits."""
    g = np.asarray(gradient)
    gmin, gmax = g.min(), g.max()
    if np.isclose(gmax, gmin):
        return g.copy(), 0.0
    delta = (gmax - gmin) / (2**num_bits - 1)
    q = np.round(g / delta) * delta
    eps_q = (delta**2) / 12.0
    return q, float(eps_q)


def hybrid_patch(clients, server, snr_map, delta_list, num_bits=8, phase_corrections=None, t_now=None):
    """
    Hybrid analog/digital patching per Algorithm 4 and Eq. 40.

    - Uses IQR-based straggler detection (not fixed threshold)
    - Threads pre-equalization phases through analog aggregation
    - Digital correction for stragglers only

    AMSE bound: (w_a²) * amse_analog + (w_d²) * eps_q

    Args:
        clients: List of client nodes
        server: Server node
        snr_map: Dict mapping client -> {server: SNR}
        delta_list: List of cascaded errors per tier
        num_bits: Quantization bits for digital path
        phase_corrections: Dict mapping client_id -> pre_equalization_phase (radians)
        t_now: Current time (skyfield Time object)

    Returns:
        tuple: (g_hybrid, amse_bound, meta)
    """
    # Remove detect_low_snr_clients - use IQR-based straggler detection only
    # Per Algorithm 4: stragglers detected from analog aggregation majority set

    g_majority, amse_analog, stragglers, majority = aircomp_aggregate(
        clients, server, snr_map, delta_list, phase_corrections, t_now
    )
    K = len(clients)
    if K == 0:
        return np.array([]), 0.0, {"stragglers": [], "majority": []}

    # For stragglers, we use digital path with quantization
    # The phase corrections for stragglers are not needed since they don't participate in analog aggregation
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


def detect_low_snr_clients(clients, server, snr_map, threshold=3.0):
    """DEPRECATED: Use identify_stragglers() instead.

    Kept for backward compatibility but should not be used.
    """
    low_clients = []

    for c in clients:
        snr = snr_map[c][server]

        if snr < threshold:
            low_clients.append(c)

    return low_clients