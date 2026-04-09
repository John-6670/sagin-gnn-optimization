from typing import List
from optimization.objective import weighted_compound_loss
from simulation.topology.aircomp import compute_amse
from simulation.network.latency import propagation_latency
from simulation.network.models import NetworkModel
from simulation.topology.nodes import Node


def greedy_placement(
    candidates: List[Node],
    clients: List[Node],
    network_model: NetworkModel,
    budget: int = 5,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> List[Node]:
    """Greedy server placement based on a weighted latency-AMSE objective."""
    selected: List[Node] = []
    remaining_budget = budget

    while remaining_budget > 0 and candidates:
        best_server = None
        best_score = float("inf")

        for server in candidates:
            total_latency = sum(propagation_latency(server, client) for client in clients)
            server_amse = compute_amse(server, clients, network_model)
            score = weighted_compound_loss(total_latency, server_amse, alpha, beta)

            if score < best_score:
                best_score = score
                best_server = server

        if best_server is None:
            break

        selected.append(best_server)
        candidates.remove(best_server)
        remaining_budget -= 1

    return selected
