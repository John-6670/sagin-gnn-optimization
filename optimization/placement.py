from typing import List

from simulation.topology.nodes import Node
from optimization.objective import compute_utility


def greedy_server_selection(
    candidates,   # list of Node
    clients,             # list of Node
    budget,              # max number of servers
    cost: dict[Node, float],
    thresh,
    alpha, beta, delta_list
) -> List[Node]:
    snr_map = {
        client: {
            server: client.compute_snr_to(server)
            for server in candidates
        }
        for client in clients
    }
    best_snr = {client: 0.0 for client in clients}

    S = []
    total_cost = 0
    candidates_cp = candidates.copy()

    while candidates_cp:
        best_server = None
        best_gain = -float("inf")

        current_utility = compute_utility(S, clients, alpha, beta, delta_list)

        for server in candidates_cp:
            if server in S:
                continue

            new_S = S + [server]
            new_utility = compute_utility(new_S, clients, alpha, beta, delta_list)

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

        candidates_cp.remove(best_server)
    return S
