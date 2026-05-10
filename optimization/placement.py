import numpy as np
from typing import List, Dict

from simulation.topology.nodes import Node
from optimization.objective import compute_utility


def predictive_ota_control(selected_servers: List[Node], clients: List[Node], t_now, target_snr=10.0) -> Dict:
    """
    Implementation of Algorithm 3: Hierarchical OTA Control (Inner Loop).
    Optimizes transmission power and phase to minimize instantaneous AMSE.
    """
    ota_results = {}
    
    for server in selected_servers:
        # Step 1: Predictive Synchronization
        # In a full simulation, timing windows are calculated based on distance/light-speed delay.
        control_params = {}
        
        for client in clients:
            # Step 2 & 3: Channel Phase Estimation & Pre-equalization
            try:
                # Uses the time-varying channel from Step 1 if available
                h = client.get_channel(server, t_now)
                snr_val = client.compute_snr_to(server, t_now)
            except TypeError:
                # Fallback to static channel if t_now is not yet fully integrated in nodes.py
                h = client.get_channel(server)
                snr_val = client.compute_snr_to(server)
                
            channel_phase = np.angle(h)
            
            # Phase pre-equalization ensures coherent signal addition at the aggregator
            pre_equalization_phase = -channel_phase
            
            # Step 4: Predictive Power Control
            # Calculates p_k^* = min(P_max, target / SNR)
            optimized_power = min(client.power, target_snr / (snr_val + 1e-12))
            
            control_params[client.id] = {
                "power": optimized_power,
                "phase": pre_equalization_phase,
                "snr": snr_val
            }
            
        ota_results[server.id] = control_params
        
    return ota_results


def greedy_server_selection(
    candidates,          # list of Node
    clients,             # list of Node
    budget,              # max number of servers
    cost: dict[Node, float],
    thresh,
    alpha, beta, delta_list,
    t_now=None           # Added to support dynamic environment snapshots
) -> List[Node]:
    """
    Outer Loop Baseline: Adapts placement to slow-timescale orbital patterns.
    """
    snr_map = {}
    for client in clients:
        snr_map[client] = {}
        for server in candidates:
            # Check if dynamic time-based SNR is supported, otherwise use static
            if t_now is not None:
                try:
                    snr_map[client][server] = client.compute_snr_to(server, t_now)
                except TypeError:
                    snr_map[client][server] = client.compute_snr_to(server)
            else:
                snr_map[client][server] = client.compute_snr_to(server)

    best_snr = {client: 0.0 for client in clients}

    S = []
    total_cost = 0
    candidates_cp = candidates.copy()

    while candidates_cp:
        best_server = None
        best_gain = -float("inf")

        current_utility = compute_utility(S, clients, alpha, beta, delta_list, snr_map=snr_map)

        for server in candidates_cp:
            if server in S:
                continue

            new_S = S + [server]
            new_utility = compute_utility(new_S, clients, alpha, beta, delta_list, snr_map=snr_map)

            gain = (current_utility - new_utility) / cost[server]
            if gain > best_gain:
                best_gain = gain
                best_server = server

        if best_server is None:
            break

        delta_snr = 0.0
        for client in clients:
            new_snr = snr_map[client][best_server]
            improvement = max(0.0, new_snr - best_snr[client])
            delta_snr += improvement
        
        if delta_snr > thresh and total_cost+cost[best_server] <= budget:
            S.append(best_server)
            total_cost += cost[best_server]
            # Update best_snr for the next iteration
            for client in clients:
                best_snr[client] = max(best_snr[client], snr_map[client][best_server])
            if total_cost >= budget:
                break

        candidates_cp.remove(best_server)
        
    return S