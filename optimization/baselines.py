import numpy as np

from optimization.objective import compute_amse_kn_from_snr
from optimization.placement import greedy_server_selection, dr_greedy_server_selection
from simulation.topology.nodes import NodeType


def lop_selection(candidates, clients, budget, cost, thresh, alpha, beta, delta_list, N, t_now=None):
    selected = []
    total_cost = 0
    C = candidates.copy()
    
    while C:
        best = None
        best_score = float('inf')
        
        for s in C:
            total_latency = sum(
                min([c.get_latency_to(ss) for ss in selected + [s]])
                for c in clients
            )
            if total_cost + cost[s] <= budget and total_latency < best_score:
                best = s
                best_score = total_latency
        
        if best is None:
            break
        
        C.remove(best)
        total_cost += cost[best]
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
    if not clients:
        return []

    def average_snr(server):
        return np.mean([server.compute_snr_to(c, t_now) for c in clients])

    C = [s for s in candidates if average_snr(s) >= thresh]
    total_cost = 0
    selected = []
    rng = np.random.default_rng(42)

    while C:
        s = rng.choice(C)
        if total_cost + cost[s] <= budget:
            selected.append(s)
            total_cost += cost[s]
        C.remove(s)

    return selected


def da_selection(candidates, clients, budget, cost, thresh, alpha, beta, delta_list, N, t_now=None):
    selected = []
    total_cost = 0
    C = candidates.copy()

    while C:
        best_candidate = None
        best_score = float('inf')

        for s in C:
            if total_cost + cost[s] > budget:
                continue

            total_loss = 0.0
            candidate_servers = selected + [s]
            for c in clients:
                best_cost = float('inf')
                for s2 in candidate_servers:
                    latency = c.get_latency_to(s2, t_now)
                    snr = c.compute_snr_to(s2, t_now)
                    amse = compute_amse_kn_from_snr(c, snr, delta_list, server=s2)
                    best_cost = min(best_cost, latency + beta * amse)
                total_loss += best_cost

            if total_loss < best_score:
                best_score = total_loss
                best_candidate = s

        if best_candidate is None:
            break

        selected.append(best_candidate)
        total_cost += cost[best_candidate]
        C.remove(best_candidate)

    return selected


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
        N=N,
        **kwargs,
    )
