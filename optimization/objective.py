import logging
import numpy as np

from simulation.topology.nodes import NodeType

logger = logging.getLogger(__name__)


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


def _compute_objective_core(servers, clients, alpha, beta, delta_list, snr_map=None, latency_map=None):
    """
    Core composite objective computation per Eq. 7:
    Cost(S) = sum_{k} min_{n in S} [alpha * L_kn + beta * AMSE_kn]
    Lower is better (cost to minimize).
    """
    if not servers:
        # Return a large cost for empty placement
        return float(len(clients) * 1000.0)

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
                      fl_result=None, use_ota=True, latency_scale=None, amse_scale=None):
    """
    Composite objective per Eq. 7 - MINIMIZE this value.
    Returns: total cost (lower = better)

    Note: Backward compatibility alias - calls _compute_objective_core
    """
    return _compute_objective_core(servers, clients, alpha, beta, delta_list, snr_map, latency_map)


def compute_placement_utility(servers, clients, alpha, beta, delta_list, snr_map=None, latency_map=None):
    """
    Placement utility U(S) per Eq. 19.
    U(S) = -Cost(S) where Cost(S) = sum_k min_{n in S} [alpha * L_kn + beta * AMSE_kn]
    Returns: utility (higher = better = more reduction from empty set)
    """
    if not servers:
        return 0.0  # U(∅) = 0 by definition

    cost = _compute_objective_core(servers, clients, alpha, beta, delta_list, snr_map, latency_map)
    return -cost


def compute_marginal_gain(servers, candidate, clients, alpha, beta, delta_list, snr_map=None, latency_map=None):
    """
    Marginal utility gain per Eq. 23 for candidate v added to set S:
    Δ(S, v) = Cost(S) - Cost(S ∪ {v}) = U(S ∪ {v}) - U(S)
    Returns: positive = improvement, negative = degradation
    """
    cost_without = _compute_objective_core(servers, clients, alpha, beta, delta_list, snr_map, latency_map)
    cost_with = _compute_objective_core(servers + [candidate], clients, alpha, beta, delta_list, snr_map, latency_map)
    return cost_without - cost_with
