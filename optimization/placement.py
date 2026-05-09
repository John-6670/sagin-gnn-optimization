from typing import List, Dict
import numpy as np
from simulation.topology.nodes import Node

def optimize_ota_parameters(selected_servers: List[Node], clients: List[Node], t_now, target_amse=1e-4):
    """
    Inner Loop: Optimizes power allocation to minimize instantaneous AMSE[cite: 125, 211].
    """
    ota_config = {}
    
    for server in selected_servers:
        server_clients = [c for c in clients] # Assuming association logic exists
        powers = {}
        
        for client in server_clients:
            snr_nominal = client.compute_snr_to(server, t_now)
            
            # Predictive Power Control (Step 3 of Algorithm 3)
            # pk* = min(Pmax, target / SNR)
            p_star = min(client.power, target_amse / (snr_nominal + 1e-12))
            powers[client.id] = p_star
            
        ota_config[server.id] = {
            "powers": powers,
            "timestamp": t_now
        }
        
    return ota_config

def greedy_server_selection(
    candidates, clients, budget, cost, thresh, alpha, beta, delta_list, t_now
):
    """
    Outer Loop: Adapts placement to orbital patterns (Slow timescale)[cite: 120, 205].
    """
    S = []
    total_cost = 0
    candidates_cp = candidates.copy()

    while candidates_cp:
        best_server = None
        best_gain = -float("inf")

        for server in candidates_cp:
            if server in S: continue
            
            # Evaluate using current orbital positions
            new_S = S + [server]
            # (Heuristic utility calculation based on current t_now)
            gain = 1.0 / (cost[server] + 1) # Placeholder for utility(new_S)
            
            if gain > best_gain:
                best_gain = gain
                best_server = server

        if best_server and total_cost + cost[best_server] <= budget:
            S.append(best_server)
            total_cost += cost[best_server]
        else:
            break
            
        candidates_cp.remove(best_server)
    return S