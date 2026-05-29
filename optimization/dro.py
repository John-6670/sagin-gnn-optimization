from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from optimization.objective import compute_utility
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScenarioBundle:
    scenarios: List[Dict]
    clients: List
    candidates: List
    delta_list: List[float]
    alpha: float
    beta: float
    latency_map: Dict = None


def sample_snr_scenarios(clients, candidates, t_now, N: int, coherence_time: float, sigma_snr: float, dt_seconds: float = 10.0):
    """Sample N SNR scenario maps with OU temporal perturbation per link."""
    rho = np.exp(-dt_seconds / max(coherence_time, 1e-6))
    scenarios = []
    logger.info("Sampling %d SNR scenarios (clients=%d, candidates=%d)", N, len(clients), len(candidates))
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
        # periodic progress log to show long-running sampling
        if N >= 10 and len(scenarios) % max(1, N // 10) == 0:
            logger.info("  sampled %d/%d SNR scenarios...", len(scenarios), N)
    return scenarios


def _cvar(values: Sequence[float], alpha_cvar: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    # remove NaN/inf to avoid invalid arithmetic
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    q = np.quantile(arr, alpha_cvar)
    tail = arr[arr <= q]
    return float(np.mean(tail)) if len(tail) > 0 else float(q)


def robust_marginal_gain(S, v, scenarios: ScenarioBundle, epsilon: float, alpha_cvar: float):
    scenario_gains = []
    for idx, snr_map in enumerate(scenarios.scenarios, start=1):
        curr = compute_utility(
            S,
            scenarios.clients,
            scenarios.alpha,
            scenarios.beta,
            scenarios.delta_list,
            snr_map=snr_map,
            latency_map=getattr(scenarios, 'latency_map', None),
        )

        new = compute_utility(
            S + [v],
            scenarios.clients,
            scenarios.alpha,
            scenarios.beta,
            scenarios.delta_list,
            snr_map=snr_map,
            latency_map=getattr(scenarios, 'latency_map', None),
        )

        gain = curr - new
        scenario_gains.append(gain)
        if len(scenarios.scenarios) >= 10 and idx % max(1, len(scenarios.scenarios)//10) == 0:
            logger.debug("  robust_marginal_gain progress: computed gains for %d/%d scenarios", idx, len(scenarios.scenarios))

    gains = np.asarray(scenario_gains, dtype=float)
    # sanitize infinities and NaNs: treat inf as large finite, NaN as zero gain
    if gains.size == 0:
        return 0.0, {'mean_gain': 0.0, 'cvar': 0.0}
    gains = gains.copy()
    # Replace NaN with 0 (no gain) and clip infinities
    nan_mask = np.isnan(gains)
    gains[nan_mask] = 0.0
    pos_inf = np.isposinf(gains)
    neg_inf = np.isneginf(gains)
    LARGE = 1e12
    gains[pos_inf] = LARGE
    gains[neg_inf] = -LARGE

    mean_gain = float(np.mean(gains))
    std_gain = float(np.std(gains))

    cvar = _cvar(gains.tolist(), alpha_cvar)
    penalty = epsilon * std_gain

    robust_gain = cvar - penalty
    
    if S:
        if getattr(scenarios, 'latency_map', None) is not None:
            lat_vals = [scenarios.latency_map.get(c, {}).get(v, c.get_latency_to(v)) for c in scenarios.clients]
        else:
            lat_vals = [c.get_latency_to(v) for c in scenarios.clients]
        avg_lat = np.mean(lat_vals)
        # Reduce latency penalty from 0.3 to 0.05 to avoid over-penalizing high-latency servers
        latency_penalty = 0.05 * avg_lat
        robust_gain -= latency_penalty
        logger.debug("  latency_penalty=%.6f applied (avg_lat=%.6f)", latency_penalty, avg_lat)
    
    if robust_gain < 1e-6 and mean_gain > 0:
        robust_gain = mean_gain * 0.7

    return float(robust_gain), {
        'mean_gain': mean_gain,
        'cvar': float(cvar),
    }


def local_one_swap(S, candidates, budget, cost, scenarios: ScenarioBundle, epsilon, alpha_cvar,
                    max_iter: int = 20, tol: float = 1e-6):
    S = list(S)
    iter_no = 0
    seen_configs = set()
    while iter_no < max_iter:
        iter_no += 1
        logger.info("local_one_swap iteration %d: current set size=%d", iter_no, len(S))

        key = tuple(sorted(getattr(x, 'id', id(x)) for x in S))
        if key in seen_configs:
            logger.info("local_one_swap: repeated configuration detected, stopping to avoid cycle")
            break
        seen_configs.add(key)

        best_swap = None
        best_improvement = tol

        for s in list(S):
            base_set = [x for x in S if x != s]
            base_cost = sum(cost[x] for x in base_set)
            old_score = robust_marginal_gain(base_set, s, scenarios, epsilon, alpha_cvar)[0]
            for v in candidates:
                if v in S:
                    continue
                if base_cost + cost[v] > budget:
                    continue
                new_score = robust_marginal_gain(base_set, v, scenarios, epsilon, alpha_cvar)[0]
                improvement = new_score - old_score
                if improvement > best_improvement:
                    best_swap = (s, v, old_score, new_score, improvement)

        if best_swap is None:
            logger.info("local_one_swap: no improving swap found, stopping at iteration %d", iter_no)
            break

        s, v, old_score, new_score, improvement = best_swap
        logger.info(
            "  best swap: replace %s with %s (old_score=%.6f new_score=%.6f delta=%.6f)",
            getattr(s, 'id', s), getattr(v, 'id', v), old_score, new_score, improvement,
        )
        S = [x for x in S if x != s] + [v]

    if iter_no >= max_iter:
        logger.info("local_one_swap: reached max_iter=%d, stopping", max_iter)

    return S


def amse_cvar_from_scenarios(S, scenarios: ScenarioBundle, alpha_cvar: float) -> float:
    """CVaR of AMSE over scenario bundle for current placement S."""
    from simulation.topology.aircomp import compute_amse_n
    amse_vals = []
    for snr_map in scenarios.scenarios:
        per_server = []
        for v in S:
            snr_dict = {c: snr_map[c].get(v, 1e-12) for c in scenarios.clients}
            sigma2 = float(np.mean([c.noise_variance for c in scenarios.clients]))
            d = scenarios.clients[0].gradient_dim if scenarios.clients else 100
            per_server.append(compute_amse_n(snr_dict, sigma2, d))
        amse_vals.append(float(np.mean(per_server)) if per_server else 0.0)
    return _cvar(amse_vals, alpha_cvar)


def bisect_lambda_for_amse_target(S, scenarios: ScenarioBundle,
                                    alpha_cvar: float, tau_amse: float) -> float:
    """
    Bisection on dual variable λ to enforce CVaR(AMSE) ≤ τ_AMSE. 
    Returns λ*.
    """
    if not S or amse_cvar_from_scenarios(S, scenarios, alpha_cvar) <= tau_amse:
        return 0.0
    lo, hi = 0.0, 1e3
    logger.info("bisect_lambda_for_amse_target: tau_amse=%.6f starting bisection", tau_amse)
    for i in range(40):
        mid = (lo + hi) / 2.0
        penalised = amse_cvar_from_scenarios(S, scenarios, alpha_cvar) / (1.0 + mid)
        logger.debug("  bisection iter %d: mid=%.6f penalised=%.6f", i+1, mid, penalised)
        if penalised > tau_amse:
            lo = mid
        else:
            hi = mid
    res = (lo + hi) / 2.0
    logger.info("bisect_lambda_for_amse_target: finished lambda=%.6f", res)
    return res
