import numpy as np
from collections import defaultdict

from simulation.topology.nodes import Node, NodeType


def _num_hops(server: Node) -> int:
    """Number of aggregation hops L(k,n) by server tier."""
    if server.type == NodeType.GROUND:
        return 1
    if server.type == NodeType.UAV:
        return 2
    return 3  # SATELLITE


def build_aggregation_tree(clients, servers, t_now=None):
    """
    Build hierarchical aggregation tree per paper Eq. 5.

    Hierarchy (bottom-up):
    - Tier 1: Clients → UAVs/HAPs (if available) or Ground BS
    - Tier 2: UAVs/HAPs → Satellites (if available) or Ground BS
    - Tier 3: Satellites → Ground Gateway (if available) or final aggregation point

    Returns:
        dict: {tier: {aggregator_id: [client_ids]}} mapping tier to aggregator-client groups
    """
    tree = defaultdict(lambda: defaultdict(list))

    # For each client, find its path to the final server
    for client in clients:
        # Find the best server (closest by latency or SNR)
        best_server = min(servers, key=lambda s: client.get_latency_to(s, t_now))

        # Build path based on server type
        path = [client]

        if best_server.type == NodeType.GROUND:
            # Direct: Client → Ground
            path.append(best_server)

        elif best_server.type == NodeType.UAV:
            # Client → UAV → Ground (or Satellite)
            path.append(best_server)
            # Find next hop
            next_hops = [s for s in servers if s.type in (NodeType.SATELLITE, NodeType.GROUND)]
            if next_hops:
                next_hop = min(next_hops, key=lambda s: best_server.get_latency_to(s, t_now))
                path.append(next_hop)

        elif best_server.type == NodeType.SATELLITE:
            # Client → Satellite → Ground
            path.append(best_server)
            # Find ground gateway
            ground_servers = [s for s in servers if s.type == NodeType.GROUND]
            if ground_servers:
                ground_gw = min(ground_servers, key=lambda s: best_server.get_latency_to(s, t_now))
                path.append(ground_gw)

        # Assign to tiers
        for tier_idx, aggregator in enumerate(path[1:], 1):
            tree[tier_idx][aggregator.id].append(client.id)

    return tree


def compute_sync_error(tier, tier_config=None):
    """
    Compute synchronization error ε_sync^(ℓ) per tier.

    Default sync errors (can be configured):
    - Tier 1 (Client→UAV/Ground): 1e-9
    - Tier 2 (UAV→Sat/Ground): 5e-9
    - Tier 3 (Sat→Ground): 1e-8
    """
    default_errors = {1: 1e-9, 2: 5e-9, 3: 1e-8}
    if tier_config and tier in tier_config:
        return tier_config[tier]
    return default_errors.get(tier, 1e-8)


# AMSE(k, n) - per client per server (single hop)
def compute_amse_kn(client: Node, server: Node, delta_list, t_now=None):
    """
    Compute AMSE for single hop (client to immediate aggregator).
    This is the building block for hierarchical AMSE.
    """
    if t_now is not None:
        snr = client.compute_snr_to(server, t_now)
    else:
        snr = client.compute_snr_to(server)

    if snr <= 0.0:
        return float("inf")

    # Single hop uses first delta
    cascaded_error = 1.0 + (delta_list[0] if delta_list else 0.0)
    return (client.noise_variance * client.gradient_dim / snr) * cascaded_error * 1e-9


# AMSE(n) - Hierarchical per Eq. 5
def compute_amse_n(snr_dict, sigma2, d, sync_error=0.0):
    """
    Original single-tier AMSE for backward compatibility.
    snr_dict: dict {k: SNR(k,n)} for all clients k served by n
    sigma2: noise variance
    d: gradient dimension
    sync_error: synchronization error term
    """
    K = len(snr_dict)
    if K == 0:
        return 0.0

    inv_snr_sum = sum(1.0 / snr for snr in snr_dict.values())
    amse = (sigma2 * d / K) * inv_snr_sum + sync_error
    return amse * 1e-9


def compute_amse_hierarchical(servers, clients, delta_list, t_now=None, tier_sync_errors=None):
    """
    Compute hierarchical AMSE per Eq. 5:
    AMSE_n = Σ_{ℓ=1}^L [ (σ²d / |K_n^(ℓ)|) Σ_{k∈K_n^(ℓ)} 1/SNR_kn^(ℓ) + ε_sync^(ℓ) ]

    Args:
        servers: Selected server nodes
        clients: Client nodes
        delta_list: List of cascaded errors per tier [δ_1, δ_2, δ_3]
        t_now: Current time (skyfield Time object or None)
        tier_sync_errors: Dict mapping tier to sync error

    Returns:
        float: Total hierarchical AMSE
    """
    if not servers or not clients:
        return 0.0

    # Compute sigma2 and d
    sigma2 = np.mean([c.noise_variance for c in clients]) if clients else 1e-9
    d = clients[0].gradient_dim if clients else 100

    # Build aggregation tree
    tree = build_aggregation_tree(clients, servers, t_now)

    total_amse = 0.0

    for tier, aggregators in tree.items():
        tier_delta = delta_list[tier - 1] if tier <= len(delta_list) else 0.0
        sync_err = compute_sync_error(tier, tier_sync_errors)

        for agg_id, client_ids in aggregators.items():
            # Get aggregator node
            agg_node = next((s for s in servers if s.id == agg_id), None)
            if agg_node is None:
                continue

            # Compute SNR for each client to this aggregator at this tier
            snr_sum_inv = 0.0
            valid_count = 0

            for client_id in client_ids:
                client_node = next((c for c in clients if c.id == client_id), None)
                if client_node is None:
                    continue

                snr = client_node.compute_snr_to(agg_node, t_now)
                if snr > 1e-12:
                    snr_sum_inv += 1.0 / snr
                    valid_count += 1

            if valid_count > 0:
                # AMSE for this tier: (σ²d / K) * Σ(1/SNR) * Π(1+δ) + ε_sync
                cascaded_error = np.prod([1.0 + delta_list[i] for i in range(tier)]) if tier <= len(delta_list) else 1.0
                tier_amse = (sigma2 * d / valid_count) * snr_sum_inv * cascaded_error * 1e-9 + sync_err
                total_amse += tier_amse

    return total_amse


def compute_amse_kn_hierarchical(client: Node, servers, delta_list, t_now=None, tier_sync_errors=None):
    """
    Compute AMSE for a specific client given the hierarchical server path.
    Used for marginal gain calculations.
    """
    if not servers:
        return float("inf")

    # Find client's aggregation path
    best_server = min(servers, key=lambda s: client.get_latency_to(s, t_now))

    path = [client]
    if best_server.type == NodeType.GROUND:
        path.append(best_server)
    elif best_server.type == NodeType.UAV:
        path.append(best_server)
        next_hops = [s for s in servers if s.type in (NodeType.SATELLITE, NodeType.GROUND)]
        if next_hops:
            path.append(min(next_hops, key=lambda s: best_server.get_latency_to(s, t_now)))
    elif best_server.type == NodeType.SATELLITE:
        path.append(best_server)
        ground_servers = [s for s in servers if s.type == NodeType.GROUND]
        if ground_servers:
            path.append(min(ground_servers, key=lambda s: best_server.get_latency_to(s, t_now)))

    total_amse = 0.0
    for tier_idx, aggregator in enumerate(path[1:], 1):
        tier_delta = delta_list[tier_idx - 1] if tier_idx <= len(delta_list) else 0.0
        sync_err = compute_sync_error(tier_idx, tier_sync_errors)

        snr = client.compute_snr_to(aggregator, t_now)
        if snr > 1e-12:
            cascaded_error = np.prod([1.0 + delta_list[i] for i in range(tier_idx)]) if tier_idx <= len(delta_list) else 1.0
            total_amse += (client.noise_variance * client.gradient_dim / snr) * cascaded_error * 1e-9 + sync_err
        else:
            return float("inf")

    return total_amse
