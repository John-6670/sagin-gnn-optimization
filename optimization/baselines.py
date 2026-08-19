import numpy as np
from typing import Dict, List

from optimization.objective import compute_amse_kn_from_snr, compute_objective
from optimization.placement import greedy_server_selection, dr_greedy_server_selection
from simulation.topology.nodes import Node, NodeType


def lop_selection(candidates, clients, budget, cost, thresh, alpha, beta, delta_list, N, t_now=None):
    selected = []
    total_cost = 0
    C = candidates.copy()
    current_latency = float('inf')

    while C:
        best = None
        best_score = float('inf')

        for s in C:
            if total_cost + cost[s] > budget:
                continue
            total_latency = sum(
                min([c.get_latency_to(ss, t_now) for ss in selected + [s]])
                for c in clients
            )
            if total_latency < best_score:
                best = s
                best_score = total_latency

        if best is None:
            break

        # Stop when marginal latency reduction is negligible relative to current latency
        if selected and current_latency > 0:
            improvement = (current_latency - best_score) / current_latency
            if improvement < 0.01:  # < 1% improvement → not worth the extra server
                break

        C.remove(best)
        total_cost += cost[best]
        selected.append(best)
        current_latency = best_score

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
        N=N,
        t_now=t_now,
    )


def nrs_selection(candidates, clients, budget, cost, thresh, alpha, beta, delta_list, N, t_now=None):
    return greedy_server_selection(
        candidates=candidates,
        clients=clients,
        budget=budget,
        cost=cost,
        thresh=float("-inf"),
        alpha=alpha,
        beta=beta,
        delta_list=delta_list,
        N=N,
        t_now=t_now,
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


def fedsn_selection(candidates: List[Node], clients: List[Node], budget: float, cost: dict, **kwargs) -> List[Node]:
    """
    FedSN-inspired selection: Prioritizes LEO-like satellites with sub-structure feasibility.
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


def hsfl_selection(
    candidates: List[Node],
    clients: List[Node],
    budget: float,
    cost: Dict[Node, float],
    thresh: float = 0.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    delta_list=None,
    N: int = 16,
    t_now=None,
    **kwargs
) -> List[Node]:
    """
    HSFL (Hierarchical Split Federated Learning) baseline from arXiv:2601.13817v3
    - Emphasizes hierarchical structure (UAV edge + Sat global)
    - Device association + split-aware selection
    """
    print(f"\n[HSFL Baseline] Evaluating {len(candidates)} candidates, budget={budget}")

    if not candidates or not clients:
        return []

    # Tier priority: UAVs preferred as edge aggregators
    tier_priority = {NodeType.UAV: 3.0, NodeType.GROUND: 2.0, NodeType.SATELLITE: 1.0}

    scores = []
    for s in candidates:
        snrs = [c.compute_snr_to(s, t_now) for c in clients]
        lats = [c.get_latency_to(s, t_now) for c in clients]
        
        avg_snr = np.mean(snrs) if snrs else 0.0
        avg_lat = np.mean(lats) if lats else float('inf')
        tier_score = tier_priority.get(s.type, 1.0)
        
        # Heterogeneity proxy (higher variance = worse for aggregation)
        hetero_proxy = np.std(snrs) if len(snrs) > 1 else 1.0
        
        # Split benefit: more offloading (higher tier) is better for resource-constrained devices
        split_benefit = tier_score / 3.0
        
        utility = (0.45 * avg_snr) / (avg_lat + 1.0) * split_benefit - 0.25 * hetero_proxy
        efficiency = utility / (cost.get(s, 1.0) + 1e-6)
        
        scores.append((s, efficiency, avg_snr, avg_lat, tier_score))

    scores.sort(key=lambda x: x[1], reverse=True)

    selected = []
    total_cost = 0.0
    for s, eff, snr, lat, tier in scores:
        c_cost = cost.get(s, 1.0)
        if total_cost + c_cost <= budget:
            selected.append(s)
            total_cost += c_cost
            print(f"  → HSFL Selected {s.id} ({s.type.value}) | Eff={eff:.4f} | SNR={snr:.2f} | Lat={lat:.1f}ms")
        else:
            break

    # Local refinement (mimics paper's iterative device association)
    if len(selected) >= 1:
        selected = _hsfl_local_refine(selected, candidates, clients, budget, cost, t_now)

    print(f"[HSFL] Final: {[s.id for s in selected]} | Cost: {total_cost:.2f}/{budget}")
    return selected


def _hsfl_local_refine(selected, candidates, clients, budget, cost, t_now):
    """Iterative refinement inspired by paper's device association + resource allocation"""
    best_set = list(selected)
    # compute_objective returns cost (lower is better)
    best_cost = compute_objective(best_set, clients, 0.5, 0.5, [0.1, 0.2])

    for _ in range(4):  # limited iterations for efficiency
        improved = False
        for i, s in enumerate(best_set):
            for alt in candidates:
                if alt in best_set:
                    continue
                if cost.get(alt, 1.0) > cost.get(s, 1.0) * 1.5:  # avoid too expensive swaps
                    continue

                trial = best_set[:i] + [alt] + best_set[i+1:]
                trial_cost = sum(cost.get(x, 1.0) for x in trial)
                if trial_cost > budget:
                    continue

                util = compute_objective(trial, clients, 0.5, 0.5, [0.1, 0.2])
                if util < best_cost:   # lower cost = better
                    best_set = trial
                    best_cost = util
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best_set


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
