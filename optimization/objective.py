import logging
import numpy as np

from simulation.topology.nodes import NodeType

logger = logging.getLogger(__name__)

# Earth and orbital constants
EARTH_RADIUS_KM = 6371.0
C_MPS = 299_792_458.0


def weighted_compound_loss(latency, amse, alpha, beta):
    """Compute a weighted compound objective for placement evaluation."""
    return alpha * latency + beta * amse


def _normalize_latency_and_amse(latency, amse, latency_scale, amse_scale):
    latency_norm = latency / latency_scale if latency_scale > 0 else latency
    amse_norm = amse / amse_scale if amse_scale > 0 else amse
    return latency_norm, amse_norm


def _num_hops_from_server(server) -> int:
    """Hops L(k,n) by server node type."""
    from simulation.topology.nodes import NodeType
    if server.type == NodeType.GROUND:
        return 1
    if server.type == NodeType.UAV:
        return 2
    return 3  # SATELLITE


def _num_hops_from_server(server) -> int:
    """Hops L(k,n) by server node type."""
    from simulation.topology.nodes import NodeType
    if server.type == NodeType.GROUND:
        return 1
    if server.type == NodeType.UAV:
        return 2
    return 3  # SATELLITE


def compute_amse_kn_from_snr(client, snr, delta_list, server=None):
    """AMSE surrogate — uses tier-specific cascaded error L(k,n)."""
    if snr <= 1e-12:
        return float("inf")

    if server is not None:
        L = _num_hops_from_server(server)
        active_deltas = delta_list[:L]
    else:
        active_deltas = delta_list
    cascaded_error = np.prod([1 + d for d in active_deltas]) if active_deltas else 1.0
    return (client.noise_variance * client.gradient_dim / snr) * cascaded_error * 1e-9


def _compute_objective_core(servers, clients, alpha, beta, delta_list, snr_map=None, latency_map=None, use_hierarchical=True, t_now=None):
    """
    Core composite objective computation per Eq. 7:
    Cost(S) = sum_{k} min_{n in S} [alpha * L_kn + beta * AMSE_kn]
    Lower is better (cost to minimize).

    If use_hierarchical=True, uses hierarchical AMSE per Eq. 5.
    """
    if not servers:
        # Return a large cost for empty placement
        return float(len(clients) * 1000.0)

    # If using hierarchical AMSE, compute per-client so AMSE differentiates servers
    if use_hierarchical:
        from simulation.topology.aircomp import compute_amse_kn_hierarchical
        tier_sync_errors = {1: 1e-9, 2: 5e-9, 3: 1e-8}

        total_cost = 0.0
        for client in clients:
            best_cost = float('inf')
            for server in servers:
                if latency_map is not None:
                    latency = float(latency_map.get(client, {}).get(server, client.get_latency_to(server, t_now)))
                else:
                    latency = float(client.get_latency_to(server, t_now))

                # Per-client hierarchical AMSE for THIS server: differentiates
                # candidates and consumes the scenario snr_map when provided.
                amse_kn = compute_amse_kn_hierarchical(
                    client, [server], delta_list, t_now=t_now,
                    tier_sync_errors=tier_sync_errors, snr_map=snr_map,
                )
                cost = weighted_compound_loss(latency, amse_kn, alpha, beta)
                best_cost = min(best_cost, cost)

            total_cost += best_cost
        return total_cost

    # Original single-tier computation
    total_cost = 0.0

    for client in clients:
        best_cost = float('inf')
        for server in servers:
            if latency_map is not None:
                latency = float(latency_map.get(client, {}).get(server, client.get_latency_to(server)))
            else:
                latency = float(client.get_latency_to(server))

            if snr_map is not None:
                snr_val = max(snr_map[client].get(server, 1e-12), 1e-12)
            else:
                snr_val = max(client.compute_snr_to(server), 1e-12)

            amse_kn = compute_amse_kn_from_snr(client, snr_val, delta_list, server=server)
            cost = weighted_compound_loss(latency, amse_kn, alpha, beta)
            best_cost = min(best_cost, cost)

        total_cost += best_cost

    return total_cost


def compute_objective(servers, clients, alpha, beta, delta_list, snr_map=None, latency_map=None,
                      fl_result=None, use_ota=True, latency_scale=None, amse_scale=None,
                      use_hierarchical=True, t_now=None):
    """
    Composite objective per Eq. 7 - MINIMIZE this value.
    Returns: total cost (lower = better)

    Note: Backward compatibility alias - calls _compute_objective_core
    """
    return _compute_objective_core(servers, clients, alpha, beta, delta_list, snr_map, latency_map, use_hierarchical, t_now)


def compute_placement_utility(servers, clients, alpha, beta, delta_list, snr_map=None, latency_map=None, use_hierarchical=True, t_now=None):
    """
    Placement utility U(S) per Eq. 19.
    U(S) = -Cost(S) where Cost(S) = sum_k min_{n in S} [alpha * L_kn + beta * AMSE_kn]
    Returns: utility (higher = better = more reduction from empty set)
    """
    if not servers:
        return 0.0  # U(∅) = 0 by definition

    cost = _compute_objective_core(servers, clients, alpha, beta, delta_list, snr_map, latency_map, use_hierarchical, t_now)
    return -cost


def compute_marginal_gain(servers, candidate, clients, alpha, beta, delta_list, snr_map=None, latency_map=None, use_hierarchical=True, t_now=None):
    """
    Marginal utility gain per Eq. 23 for candidate v added to set S:
    Δ(S, v) = Cost(S) - Cost(S ∪ {v}) = U(S ∪ {v}) - U(S)
    Returns: positive = improvement, negative = degradation
    """
    cost_without = _compute_objective_core(servers, clients, alpha, beta, delta_list, snr_map, latency_map, use_hierarchical, t_now)
    cost_with = _compute_objective_core(servers + [candidate], clients, alpha, beta, delta_list, snr_map, latency_map, use_hierarchical, t_now)
    return cost_without - cost_with


# Cache for orbital average SNR computations
_orbital_avg_snr_cache = {}

def _get_orbital_cache_key(client, server):
    """Generate a cache key for client-server pair."""
    # Use client position and server TLE/position as key
    # Round client position to ~1km for cache efficiency
    client_pos = tuple(np.round(client.position * 111.0, 1))  # ~1km precision
    if server.sat_obj is not None:
        # Use satellite TLE line 1 as identifier
        sat_id = server.tle[0][:20] if server.tle else str(server.id)
    else:
        sat_id = str(server.id)
    return (client_pos, sat_id)

def compute_orbital_avg_snr(client, server, t0=None, orbital_period_s=None, num_samples=8, use_cache=True):
    """
    Compute orbital-average SNR per Eq. 21:
    𝔼[SNR_kn(t)] = (p_k/σ²) * (1/T)∫₀ᵀ ρ²_kn(t) dt * 𝔼[|h̃|²]

    For satellite links, ρ²(t) varies significantly over orbital period.
    For HAP/Ground links, we approximate as static (instantaneous ρ²).

    Args:
        client: Client node
        server: Server node (satellite, UAV, or ground)
        t0: Reference time (skyfield Time object), defaults to current time
        orbital_period_s: Orbital period in seconds (default: computed from altitude)
        num_samples: Number of time samples for numerical integration (default: 8 for speed)
        use_cache: Whether to use cached results (default: True)

    Returns:
        float: Orbital-average SNR (linear scale)
    """
    from simulation.topology.nodes import NodeType, TS

    # For non-satellite servers, use instantaneous SNR (no orbital variation)
    if server.type != NodeType.SATELLITE:
        snr_val = max(client.compute_snr_to(server, t0), 1e-12)
        return snr_val

    # Check cache
    cache_key = _get_orbital_cache_key(client, server)
    if use_cache and cache_key in _orbital_avg_snr_cache:
        return _orbital_avg_snr_cache[cache_key]

    # For satellite: numerical integration over orbital period
    if orbital_period_s is None:
        # Compute orbital period from altitude using Kepler's third law
        # T = 2π * sqrt(a³ / μ) where μ = 3.986e5 km³/s²
        alt_km = server.position[2] if len(server.position) > 2 else 550.0
        semi_major_axis = EARTH_RADIUS_KM + alt_km
        orbital_period_s = 2 * np.pi * np.sqrt(semi_major_axis**3 / 3.986e5)

    # Expected fading power for Rician fading: 𝔼[|h̃|²] = 1 (normalized)
    # For LEO-Ground link, K-factor from channel model is 10 dB
    # so 𝔼[|h̃|²] = 1 (the fading is already normalized in channel_coefficient)
    expected_fading_power = 1.0

    # Numerical integration of ρ²(t) over orbital period
    # ρ(t) = c / (4π f_c d(t)) for amplitude, so ρ²(t) = c² / (16π² f_c² d(t)²)
    # But channel_model computes rho = c/(4π f_c d) [amplitude], so |h| = rho * |h̃|
    # and |h|² = rho² * |h̃|²
    # SNR = p * |h|² / σ² = (p/σ²) * rho² * |h̃|²

    fc = 2.4e9  # carrier frequency from channel model
    p_k = server.power
    sigma2 = client.noise_variance

    # Sample over orbital period - use fewer samples for speed
    t_samples = np.linspace(0, orbital_period_s, num_samples)
    rho_squared_sum = 0.0

    # Get client's position (approximately static over one orbital period for LEO)
    # Client position at t0
    if t0 is None:
        t0 = TS.now()
    client_pos_t0 = client.get_position_at(t0)

    for dt in t_samples:
        # Create time at t0 + dt using skyfield's timescale
        # Add dt seconds to the base time
        t_sample = TS.tt_jd(getattr(t0, "tt", 0.0) + dt / 86400.0)

        if server.sat_obj is not None:
            sat_pos = server.sat_obj.at(t_sample).position.km
            d_km = np.linalg.norm(sat_pos - client_pos_t0)
            d_m = d_km * 1000.0
            rho = C_MPS / (4.0 * np.pi * fc * d_m)
            rho_squared_sum += rho * rho
        else:
            # Fallback: use instantaneous position
            sat_pos = server.get_position_at(t0)
            d_km = np.linalg.norm(sat_pos - client_pos_t0)
            d_m = d_km * 1000.0
            rho = C_MPS / (4.0 * np.pi * fc * d_m)
            rho_squared_sum += rho * rho
            # If no TLE, all samples are the same, so break
            break

    avg_rho_squared = rho_squared_sum / len(t_samples)

    # Apply weather loss (average over clear/rain states)
    # Clear: 2 dB loss, Rain: weighted avg of 3/5 dB
    # Stationary distribution: π_clear = 0.25/(0.08+0.25) ≈ 0.76, π_rain ≈ 0.24
    avg_weather_loss_lin = (0.76 * 10**(-2.0/10.0) + 0.24 * (0.7*10**(-3.0/10.0) + 0.3*10**(-5.0/10.0)))

    # Orbital-average SNR
    orbital_avg_snr = (p_k / sigma2) * avg_rho_squared * expected_fading_power * avg_weather_loss_lin

    result = max(orbital_avg_snr, 1e-12)

    # Cache result
    if use_cache:
        _orbital_avg_snr_cache[cache_key] = result

    return result


def clear_orbital_cache():
    """Clear the orbital SNR cache."""
    global _orbital_avg_snr_cache
    _orbital_avg_snr_cache.clear()


def compute_orbital_avg_pathloss_squared(client, server, t0=None, orbital_period_s=None, num_samples=20):
    """
    Compute just the orbital-average pathloss component (ρ²) for use in AMSE surrogate.
    Returns (1/T)∫₀ᵀ ρ²_kn(t) dt

    This is useful when we want to separate the pathloss from power/noise/fading.
    """
    from simulation.topology.nodes import NodeType, TS

    if server.type != NodeType.SATELLITE:
        h = client.get_channel(server, t0)
        return np.abs(h)**2 / max(1e-12, client.power / client.noise_variance)  # rho² = |h|² / (p/σ²)

    if orbital_period_s is None:
        alt_km = server.position[2] if len(server.position) > 2 else 550.0
        semi_major_axis = EARTH_RADIUS_KM + alt_km
        orbital_period_s = 2 * np.pi * np.sqrt(semi_major_axis**3 / 3.986e5)

    fc = 2.4e9

    if t0 is None:
        t0 = TS.now()
    client_pos_t0 = client.get_position_at(t0)

    t_samples = np.linspace(0, orbital_period_s, num_samples)
    rho_squared_sum = 0.0

    for dt in t_samples:
        from skyfield.api import Time
        t_sample = TS.tt_jd(getattr(t0, "tt", 0.0) + dt / 86400.0)

        if server.sat_obj is not None:
            sat_pos = server.sat_obj.at(t_sample).position.km
            d_km = np.linalg.norm(sat_pos - client_pos_t0)
            d_m = d_km * 1000.0
            rho = C_MPS / (4.0 * np.pi * fc * d_m)
            rho_squared_sum += rho * rho
        else:
            sat_pos = server.get_position_at(t0)
            d_km = np.linalg.norm(sat_pos - client_pos_t0)
            d_m = d_km * 1000.0
            rho = C_MPS / (4.0 * np.pi * fc * d_m)
            rho_squared_sum += rho * rho
            break

    return rho_squared_sum / len(t_samples)
