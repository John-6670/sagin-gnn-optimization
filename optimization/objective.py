import numpy as np

from simulation.topology.aircomp import compute_amse_kn


def weighted_compound_loss(latency, amse, alpha, beta):
    """Compute a weighted compound objective for placement evaluation."""
    return alpha * latency + beta * amse


def compute_amse_kn_from_snr(client, snr, delta_list):
    if snr <= 1e-12:
        return float("inf")

    cascaded_error = np.prod([1 + d for d in delta_list])
    return (client.noise_variance * client.gradient_dim / snr) * cascaded_error


def compute_utility(servers, clients, alpha, beta, delta_list, snr_map=None):
    if not servers:
        return 0.0
    
    total_utility = 0.0

    for client in clients:
        best_cost = float('inf')

        for server in servers:
            latency = client.get_latency_to(server)

            if hasattr(server, 'tier'):
                if server.tier == 'satellite':
                    latency *= 1.2
                elif server.tier == 'uav':
                    latency *= 1.25

            if snr_map is not None:
                snr = snr_map[client].get(server, 0.0)
                amse = compute_amse_kn_from_snr(client, snr, delta_list)
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

        if best_cost < float('inf'):
            total_utility += best_cost

    return float(total_utility)
