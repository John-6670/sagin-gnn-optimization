from simulation.topology.amse import compute_amse
from simulation.topology.snr import compute_snr

def greedy_placement(candidates, clients, budget=5):
    """
    Simplified greedy placement:
    - candidates: list of servers (sat/UAV/ground)
    - clients: list of clients (ground nodes)
    """
    selected = []
    remaining_budget = budget

    while remaining_budget > 0 and candidates:
        best_server = None
        best_gain = -float('inf')

        for server in candidates:
            amse = compute_amse(server, clients, compute_snr)
            gain = 1 / (amse + 1e-6)  # higher gain for lower AMSE
            if gain > best_gain:
                best_gain = gain
                best_server = server

        if best_server is None:
            break
        selected.append(best_server)
        candidates.remove(best_server)
        remaining_budget -= 1

    return selected