import random

from optimization.placement import greedy_server_selection
from simulation.topology.nodes import NodeType


def lop_selection(candidates, clients, budget):
    selected = []

    for _ in range(budget):
        best = None
        best_score = float("inf")

        for s in candidates:
            if s in selected:
                continue

            total_latency = sum(
                min([c.get_latency_to(ss) for ss in selected + [s]])
                for c in clients
            )

            if total_latency < best_score:
                best_score = total_latency
                best = s

        if best:
            selected.append(best)

    return selected


def go_selection(nodes, clients, budget, **kwargs):
    ground_nodes = [n for n in nodes if n.type == NodeType.GROUND]

    return greedy_server_selection(
        candidate_servers=ground_nodes,
        clients=clients,
        budget=budget,
        **kwargs
    )


def nrs_selection(candidates, clients, budget, cost, alpha, beta, delta_list):
    # same greedy but NO threshold
    return greedy_server_selection(
        candidate_servers=candidates,
        clients=clients,
        budget=budget,
        cost=cost,
        thresh=-float("inf"),  # disables threshold
        alpha=alpha,
        beta=beta,
        delta_list=delta_list
    )


def random_selection(candidates, budget):
    return random.sample(candidates, min(budget, len(candidates)))