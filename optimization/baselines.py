import random

from optimization.placement import greedy_server_selection, dr_greedy_server_selection
from simulation.topology.nodes import NodeType


def lop_selection(candidates, clients, budget, cost, thresh, alpha, beta, delta_list, N, t_now=None):
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


def go_selection(candidates, clients, budget, cost, thresh, alpha, beta, delta_list, N, t_now=None):
    ground_candidates = [n for n in candidates if n.type == NodeType.GROUND]

    return greedy_server_selection(
        candidates=ground_candidates,
        clients=clients,
        budget=budget,
        cost=cost,
        thresh=thresh,
        alpha=alpha,
        beta=beta,
        delta_list=delta_list,
        N=N
    )


def nrs_selection(candidates, clients, budget, cost, thresh, alpha, beta, delta_list, N, t_now=None):
    # same greedy but NO threshold
    return greedy_server_selection(
        candidates=candidates,
        clients=clients,
        budget=budget,
        cost=cost,
        thresh=float("-inf"),  # disables threshold
        alpha=alpha,
        beta=beta,
        delta_list=delta_list,
        N=N
    )


def random_selection(candidates, clients, budget, cost, thresh, alpha, beta, delta_list, N, t_now=None):
    return random.sample(candidates, min(budget, len(candidates)))


def dr_selection(candidates, clients, budget, cost, thresh, alpha, beta, delta_list, N, t_now=None, **kwargs):
    return dr_greedy_server_selection(
        candidates=candidates,
        clients=clients,
        budget=budget,
        cost=cost,
        thresh=thresh,
        alpha=alpha,
        beta=beta,
        delta_list=delta_list,
        t_now=t_now,
        epsilon=0.05,
        alpha_cvar=0.05,
        N=N,
        **kwargs,
    )
