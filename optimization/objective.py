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


def compute_utility(servers, clients, alpha, beta, delta_list, snr_map=None, latency_map=None, fl_result=None, use_ota=True, latency_scale=None, amse_scale=None):
    from simulation.topology.aircomp import compute_amse_n
    
    if not servers:
        return float(len(clients) * 2000.0)
    
    total_utility = 0.0
    total_amse = 0.0
    
    # Precompute per client-server latencies and per client-server AMSE proxy
    all_latencies = []
    per_pair_amse = []
    per_client_server_amse = {}
    for client in clients:
        per_client_server_amse[client] = {}
        for server in servers:
            if latency_map is not None:
                latency = float(latency_map.get(client, {}).get(server, client.get_latency_to(server)))
            else:
                latency = float(client.get_latency_to(server))
            all_latencies.append(latency)

            if snr_map is not None:
                snr_val = snr_map[client].get(server, 1e-12)
            else:
                snr_val = client.compute_snr_to(server)

            # per-client-server AMSE surrogate (kn)
            amse_kn = compute_amse_kn_from_snr(client, max(snr_val, 1e-12), delta_list, server=server)
            per_client_server_amse[client][server] = amse_kn
            per_pair_amse.append(amse_kn)

    # Normalize scales: use robust percentile to avoid extreme outliers dominating
    import numpy as _np
    latency_scale = max(float(latency_scale) if latency_scale is not None else (_np.percentile(all_latencies, 95) if all_latencies else 1.0), 1e-6)
    amse_scale = max(float(amse_scale) if amse_scale is not None else (_np.percentile(per_pair_amse, 95) if per_pair_amse else 1.0), 1e-12)

    # 2. First assignment: choose per-client best server using normalized weighted loss
    client_to_server = {}
    for client in clients:
        best_server = None
        best_cost = float('inf')
        for server in servers:
            latency = float(latency_map.get(client, {}).get(server, client.get_latency_to(server))) if latency_map is not None else float(client.get_latency_to(server))
            raw_amse = per_client_server_amse.get(client, {}).get(server, float('inf'))
            norm_lat, norm_amse = _normalize_latency_and_amse(latency, raw_amse, latency_scale, amse_scale)
            load_factor = 1.0 + 0.3 * getattr(client, 'load', 0.0)
            compound = weighted_compound_loss(norm_lat, norm_amse, alpha, beta) * load_factor
            if compound < best_cost:
                best_cost = compound
                best_server = server
        client_to_server[client] = best_server
    
    # Compute per-server aggregated AMSE only for assigned clients (final aggregated values)
    server_amse = {}
    server_amse_cost = {}
    sigma2 = 10.0  # default noise variance
    gradient_dim = clients[0].gradient_dim if clients else 100

    for server in servers:
        assigned_clients = [c for c, s in client_to_server.items() if s == server]
        if not assigned_clients:
            server_amse[server] = 0.0
            server_amse_cost[server] = 0.0
            continue
        snr_dict = {}
        for c in assigned_clients:
            if snr_map is not None:
                snr = snr_map[c].get(server, 1e-12)
            else:
                snr = c.compute_snr_to(server)
            snr_dict[c] = max(snr, 1e-12)

        server_amse[server] = compute_amse_n(snr_dict, sigma2, gradient_dim)
        server_amse_cost[server] = server_amse[server]

    # accumulate total AMSE across clients according to assignment
    for client in clients:
        best_server = client_to_server.get(client)
        if best_server is not None:
            total_amse += server_amse.get(best_server, 0.0)
    
    # Normalize and compute final utility
    latency_scale = max(float(latency_scale) if latency_scale is not None else (max(all_latencies) if all_latencies else 1.0), 1e-6)
    amse_scale = max(float(amse_scale) if amse_scale is not None else (max(all_server_amses) if all_server_amses else 1.0), 1e-12)
    
    
    total_utility = 0.0
    for client in clients:
        best_cost = float('inf')
        for server in servers:
            if latency_map is not None:
                latency = float(latency_map.get(client, {}).get(server, client.get_latency_to(server)))
            else:
                latency = client.get_latency_to(server)

            raw_amse = server_amse[server]
            score_amse = server_amse_cost[server]
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
