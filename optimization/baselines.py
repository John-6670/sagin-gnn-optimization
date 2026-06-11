import numpy as np
from typing import List

from optimization.objective import compute_amse_kn_from_snr
from optimization.placement import greedy_server_selection, dr_greedy_server_selection
from simulation.topology.nodes import Node, NodeType


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
    rng = np.random.default_rng()

    while C:
        s = rng.choice(C)
        if total_cost + cost[s] <= budget:
            selected.append(s)
            total_cost += cost[s]
        C.remove(s)

    print(
        "FedSN selected:",
        [s.id for s in selected]
    )
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


def fedsn_selection(candidates: List[Node], clients: List[Node], budget: float, cost: dict, **kwargs) -> List[Node]:
    """
    Improved FedSN-inspired selection: Prioritizes LEO-like satellites with sub-structure feasibility.
    Emulates heterogeneity by preferring nodes with better compute proxies (power) and visibility.
    """
    print(f"\n[FedSN Baseline Selection] Evaluating {len(candidates)} candidates with budget {budget}")
    if not candidates:
        return []

    t_now = kwargs.get("t_now", None)
    utility_scores = []

    for s in candidates:
        snrs = [c.compute_snr_to(s, t_now) for c in clients]
        latencies = [c.get_latency_to(s, t_now) for c in clients]
        
        avg_snr = np.mean(snrs) if snrs else 0.0
        avg_lat = np.mean(latencies) if latencies else float('inf')
        node_cost = cost.get(s, 1.0)
        # Proxy for compute capability (higher power = better for sub-structures)
        compute_proxy = getattr(s, 'power', 1.0)

        # FedSN-like score: reward high SNR, low latency, high compute, penalize cost
        comm_score = avg_snr / (1.0 + np.std(snrs) if snrs else 1.0)
        utility = 0.5 * comm_score - 0.3 * (avg_lat / 1000.0) + 0.2 * np.log1p(compute_proxy)
        efficiency = utility / (1.0 + np.log1p(node_cost))

        utility_scores.append((s, efficiency, avg_snr, avg_lat, node_cost))

    utility_scores.sort(key=lambda x: x[1], reverse=True)

    selected = []
    current_cost = 0.0
    for s, eff, snr, lat, c_cost in utility_scores:
        if current_cost + c_cost <= budget:
            selected.append(s)
            current_cost += c_cost
            print(f"  -> FedSN Selected {s.id} ({s.type.name}) | Eff: {eff:.4f} | SNR: {snr:.2f} | Lat: {lat:.1f}ms")
        else:
            print(f"  x Skipped {s.id} (cost {c_cost:.2f} exceeds remaining)")

    print(f"[FedSN] Final selection: {[s.id for s in selected]} | Total cost: {current_cost:.2f}/{budget}")
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
