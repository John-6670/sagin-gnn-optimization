from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from optimization.objective import compute_utility


@dataclass
class ScenarioBundle:
    scenarios: List[Dict]
    clients: List
    candidates: List
    delta_list: List[float]
    alpha: float
    beta: float


def sample_snr_scenarios(clients, candidates, t_now, N: int, coherence_time: float, sigma_snr: float, dt_seconds: float = 10.0):
    """Sample N SNR scenario maps with OU temporal perturbation per link."""
    rho = np.exp(-dt_seconds / max(coherence_time, 1e-6))
    scenarios = []
    for _ in range(N):
        snr_map = {}
        for c in clients:
            snr_map[c] = {}
            for s in candidates:
                base = max(c.compute_snr_to(s, t_now), 1e-12)
                noise = np.random.normal(0.0, sigma_snr)
                log_snr = np.log(base)
                perturbed_log = rho * log_snr + np.sqrt(max(0.0, 1.0 - rho * rho)) * noise
                snr_map[c][s] = float(np.exp(perturbed_log))
        scenarios.append(snr_map)
    return scenarios


def _cvar(values: Sequence[float], alpha_cvar: float) -> float:
    if len(values) == 0:
        return 0.0
    arr = np.asarray(values, dtype=float)
    q = np.quantile(arr, alpha_cvar)
    tail = arr[arr >= q]
    return float(np.mean(tail)) if tail.size else float(q)


def robust_marginal_gain(S, v, scenarios: ScenarioBundle, epsilon: float, alpha_cvar: float):
    """Worst-case CVaR gain under Wasserstein radius epsilon using dual lambda via bisection."""
    losses = []
    for snr_map in scenarios.scenarios:
        curr = compute_utility(S, scenarios.clients, scenarios.alpha, scenarios.beta, scenarios.delta_list, snr_map=snr_map)
        new = compute_utility(S + [v], scenarios.clients, scenarios.alpha, scenarios.beta, scenarios.delta_list, snr_map=snr_map)
        losses.append(curr - new)

    norms = np.abs(losses)

    def dual_obj(lam: float) -> float:
        inner = np.maximum(0.0, np.asarray(losses) - lam * norms)
        return lam * epsilon + float(np.mean(inner))

    lo, hi = 0.0, 1e3
    for _ in range(60):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if dual_obj(m1) < dual_obj(m2):
            hi = m2
        else:
            lo = m1
    lam_star = (lo + hi) / 2
    wc_cvar = _cvar(losses, alpha_cvar) - lam_star * epsilon
    return float(wc_cvar), float(lam_star)


def local_one_swap(S, candidates, budget, cost, scenarios: ScenarioBundle, epsilon, alpha_cvar):
    improved = True
    S = list(S)
    while improved:
        improved = False
        for s in list(S):
            for v in candidates:
                if v in S:
                    continue
                trial = [x for x in S if x != s] + [v]
                if sum(cost[x] for x in trial) > budget:
                    continue
                old_score = sum(robust_marginal_gain([x for x in S if x != s], s, scenarios, epsilon, alpha_cvar)[0] for _ in [0])
                new_score = sum(robust_marginal_gain([x for x in S if x != s], v, scenarios, epsilon, alpha_cvar)[0] for _ in [0])
                if new_score > old_score:
                    S = trial
                    improved = True
                    break
            if improved:
                break
    return S
