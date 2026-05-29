import numpy as np

from simulation.topology.aircomp import compute_amse_kn


def weighted_compound_loss(latency, amse, alpha, beta):
    """Compute a weighted compound objective for placement evaluation."""
    return alpha * latency + beta * amse


def _num_hops_from_server(server) -> int:
    """Hops L(k,n) by server node type."""
    from simulation.topology.nodes import NodeType
    if server.type == NodeType.GROUND:
        return 1
    if server.type == NodeType.UAV:
        return 2
    return 3  # SATELLITE


def compute_amse_kn_from_snr(client, snr, delta_list, server=None):
    """AMSE surrogate per Eq. 20 — uses tier-specific cascaded error L(k,n)."""
    if snr <= 1e-12:
        return float("inf")

    if server is not None:
        L = _num_hops_from_server(server)
        active_deltas = delta_list[:L]
    else:
        active_deltas = delta_list
    cascaded_error = np.prod([1 + d for d in active_deltas]) if active_deltas else 1.0
    return (client.noise_variance * client.gradient_dim / snr) * cascaded_error


def compute_utility(servers, clients, alpha, beta, delta_list, snr_map=None, latency_map=None):
    if not servers:
        # Return a large finite penalty when no servers are available so that
        # adding any feasible server reduces the objective (monotonicity),
        # while avoiding infinities that lead to NaN arithmetic elsewhere.
        return float(len(clients) * 1e9) if clients else 1e9
    
    total_utility = 0.0

    for client in clients:
        best_cost = float('inf')

        for server in servers:
            if latency_map is not None:
                latency = float(latency_map.get(client, {}).get(server, client.get_latency_to(server)))
            else:
                latency = client.get_latency_to(server)

            if hasattr(server, 'tier'):
                if server.tier == 'satellite':
                    latency *= 1.2
                elif server.tier == 'uav':
                    latency *= 1.25

            if snr_map is not None:
                snr = snr_map[client].get(server, 0.0)
                amse = compute_amse_kn_from_snr(client, snr, delta_list, server=server)
            else:
                amse = compute_amse_kn(client, server, delta_list)

            routing_penalty = 0.0
    
            if hasattr(server, 'hop_count'):
                routing_penalty += 0.05 * server.hop_count

            compound = weighted_compound_loss(
                latency,
                amse,
                alpha,
                beta,
            )

            total_cost = compound + routing_penalty

            best_cost = min(best_cost, total_cost)

        total_utility += best_cost if best_cost < float("inf") else 1e9

    return total_utility
