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


def compute_utility(servers, clients, alpha, beta, delta_list, snr_map=None, latency_map=None, fl_result=None, use_ota=True):
    from simulation.topology.aircomp import compute_amse_n
    
    if not servers:
        return float(len(clients) * 2000.0)
    
    total_utility = 0.0
    total_amse = 0.0
    
    # Precompute per-server aggregated AMSE
    server_amse = {}
    server_amse_transformed = {}
    
    for server in servers:
        snr_dict = {}
        for client in clients:
            if snr_map is not None:
                snr = snr_map[client].get(server, 1e-12)
            else:
                snr = client.compute_snr_to(server)
            snr_dict[client] = max(snr, 1e-12)
        
        # Compute aggregated AMSE for this server across all clients
        sigma2 = 10.0  # default noise variance
        gradient_dim = clients[0].gradient_dim if clients else 100
        server_amse[server] = compute_amse_n(snr_dict, sigma2, gradient_dim)
        # transform AMSE to a log-scale score so multiplicative gaps become additive
        eps = 1e-18
        server_amse_transformed[server] = -np.log10(server_amse[server] + eps)
    
    # Collect latencies per client-server pair for normalization
    all_latencies = []
    # use transformed AMSE values for normalization and scoring
    all_server_amses = list(server_amse_transformed.values())
    
    for client in clients:
        best_cost = float('inf')
        best_amse = float('inf')
        best_server = None

        for server in servers:
            if latency_map is not None:
                latency = float(latency_map.get(client, {}).get(server, client.get_latency_to(server)))
            else:
                latency = client.get_latency_to(server)

            all_latencies.append(latency)
            raw_amse = server_amse[server]
            score_amse = server_amse_transformed[server]

            load_factor = 1.0 + 0.3 * getattr(client, 'load', 0.0)
            # quick heuristic selection before normalization uses transformed score
            compound = (latency, score_amse, load_factor)

            if compound[0] + compound[1] * 0.1 < best_cost:
                best_cost = compound[0] + compound[1] * 0.1
                best_amse = raw_amse
                best_server = server

        total_amse += best_amse
    
    # Normalize and compute final utility
    latency_scale = max(max(all_latencies) if all_latencies else 1.0, 1e-6)
    amse_scale = max(max(all_server_amses) if all_server_amses else 1.0, 1e-12)
    
    
    total_utility = 0.0
    for client in clients:
        best_cost = float('inf')
        for server in servers:
            if latency_map is not None:
                latency = float(latency_map.get(client, {}).get(server, client.get_latency_to(server)))
            else:
                latency = client.get_latency_to(server)

            raw_amse = server_amse[server]
            score_amse = server_amse_transformed[server]
            normalized_latency = latency / latency_scale
            normalized_amse = score_amse / amse_scale

            load_factor = 1.0 + 0.3 * getattr(client, 'load', 0.0)
            compound = weighted_compound_loss(normalized_latency, normalized_amse, alpha, beta) * load_factor

            best_cost = min(best_cost, compound)

        total_utility += best_cost
    
    gamma = 0.6
    fl_penalty = gamma * total_amse * 0.25
    
    avg_amse = total_amse / len(clients) if clients else 0.0
    if fl_result is not None:
        final_loss = fl_result.get('final_loss', 0.0)
        mean_amse = fl_result.get('mean_amse', avg_amse)
        fl_penalty += 0.8 * final_loss + 0.5 * mean_amse

    final_utility = total_utility + fl_penalty
    
    return final_utility
