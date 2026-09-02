from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from optimization.objective import compute_objective
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
    latency_scale: float = None
    amse_scale: float = None
    t_now: object = None
    use_hierarchical: bool = True


def sample_snr_scenarios(clients, candidates, t_now, N: int, coherence_time: float, sigma_snr: float, dt_seconds: float = 10.0, use_orbital_avg: bool = False):
    """Sample N SNR scenario maps with OU temporal perturbation per link.

    If use_orbital_avg=True (for outer loop placement), satellite links use orbital-average SNR
    per Eq. 21. Non-satellite links use instantaneous SNR.
    """
    from simulation.topology.nodes import NodeType
    from optimization.objective import compute_orbital_avg_snr

    rho = np.exp(-dt_seconds / max(coherence_time, 1e-6))
    scenarios = []
    logger.info("Sampling %d SNR scenarios (clients=%d, candidates=%d, orbital_avg=%s)", N, len(clients), len(candidates), use_orbital_avg)
    for _ in range(N):
        snr_map = {}
        for c in clients:
            snr_map[c] = {}
            for s in candidates:
                if use_orbital_avg and s.type == NodeType.SATELLITE:
                    # Use orbital-average SNR for satellite links (Eq. 21)
                    base = compute_orbital_avg_snr(c, s, t_now)
                else:
                    # Instantaneous SNR for non-satellite or inner loop
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


def _cvar(values: Sequence[float], alpha_cvar: float, tail: str = 'lower') -> float:
    """
    Compute CVaR per Eq. 15: CVaR_α(L) = min_t { t + 1/(1-α) E[(L-t)⁺] }
    For worst 5% tail: α = 0.95 (upper tail).
    """
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    if tail == 'lower':
        q = np.quantile(arr, alpha_cvar)
        tail_vals = arr[arr <= q]
    else:
        alpha_cvar = 1.0 - alpha_cvar  # For upper tail (worst cases)
        q = np.quantile(arr, alpha_cvar)
        tail_vals = arr[arr >= q]
        # Empirical CVaR for upper tail (Eq. 15)
        return float(np.mean(tail_vals)) if len(tail_vals) > 0 else float(q)
    return float(np.mean(tail_vals)) if len(tail_vals) > 0 else float(q)


def compute_cvar_empirical(loss_history, alpha=0.95):
    """
    Empirical CVaR per Eq. 15: CVaR_α(L) = mean of worst (1-α) fraction.
    For worst 5% tail: α = 0.95.
    """
    arr = np.asarray(loss_history, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    arr_sorted = np.sort(arr)
    k = max(1, int(np.ceil((1 - alpha) * len(arr))))
    tail = arr_sorted[-k:]
    return float(np.mean(tail))


def snr_ground_metric(xi1, xi2, weights=None):
    """
    Ground metric for Wasserstein distance in SNR space.
    Weighted L2 on log-SNR vectors (Eq. 27 uses ||u|| ≤ λ in dual space).
    xi: flattened SNR vector for (client, server) pairs.
    """
    if weights is None:
        weights = np.ones_like(xi1)
    return np.sqrt(np.sum(weights * (np.log(np.maximum(xi1, 1e-12)) - np.log(np.maximum(xi2, 1e-12)))**2))


def wasserstein_dual_objective(lambda_dual, scenario_values, epsilon, ground_metric=None):
    """
    Wasserstein dual objective per Eq. 27:
    inf_λ≥0 { λε + (1/N) Σ_i sup_ξ [value(ξ) - λ d(ξ, ξ_i)] }

    For linear value function and quadratic ground metric, the sup has closed form.
    """
    N = len(scenario_values)
    if ground_metric is None:
        # Simple approximation: sup over ξ is shifted by lambda
        return lambda_dual * epsilon + np.mean(scenario_values) - lambda_dual * np.std(scenario_values)

    # For each scenario i, sup_ξ [value(ξ) - λ * d(ξ, ξ_i)]
    # Under linear value and quadratic metric, this is approximately value_i + λ * sensitivity
    # We use a practical approximation: mean - λ * std
    mean_val = np.mean(scenario_values)
    std_val = np.std(scenario_values)
    return lambda_dual * epsilon + mean_val - lambda_dual * std_val


def robust_marginal_gain(S, v, scenarios: ScenarioBundle, epsilon: float, alpha_cvar: float):
    """
    Compute robust marginal gain using Wasserstein DRO per Eq. 27.

    The DRO formulation: min_λ≥0 { λε + (1/N) Σ_i sup_ξ [gain(ξ) - λ d(ξ, ξ_i)] }
    For linear gain in ξ and quadratic ground metric, this has a closed form.
    """
    scenario_gains = []
    scenario_latency_deltas = []
    scenario_snr_vectors = []

    for idx, snr_map in enumerate(scenarios.scenarios, start=1):
        curr = compute_objective(
            S,
            scenarios.clients,
            scenarios.alpha,
            scenarios.beta,
            scenarios.delta_list,
            snr_map=snr_map,
            latency_map=getattr(scenarios, 'latency_map', None),
            latency_scale=getattr(scenarios, 'latency_scale', None),
            amse_scale=getattr(scenarios, 'amse_scale', None),
            t_now=getattr(scenarios, 't_now', None),
            use_hierarchical=getattr(scenarios, 'use_hierarchical', True),
        )

        new = compute_objective(
            S + [v],
            scenarios.clients,
            scenarios.alpha,
            scenarios.beta,
            scenarios.delta_list,
            snr_map=snr_map,
            latency_map=getattr(scenarios, 'latency_map', None),
            latency_scale=getattr(scenarios, 'latency_scale', None),
            amse_scale=getattr(scenarios, 'amse_scale', None),
            t_now=getattr(scenarios, 't_now', None),
            use_hierarchical=getattr(scenarios, 'use_hierarchical', True),
        )

        gain = curr - new
        scenario_gains.append(gain)

        # Collect SNR vectors for ground metric calculation
        snr_vec = []
        for c in scenarios.clients:
            snr_vec.append(max(snr_map[c].get(v, 1e-12), 1e-12))
        scenario_snr_vectors.append(np.array(snr_vec))

        # compute average min-latency before/after to capture latency impact
        lat_map = getattr(scenarios, 'latency_map', None)
        t_now = getattr(scenarios, 't_now', None)
        curr_lats = []
        new_lats = []
        for c in scenarios.clients:
            # current config
            if S:
                lat_before = min([
                    float(lat_map.get(c, {}).get(s, c.get_latency_to(s, t_now))) if lat_map is not None else float(c.get_latency_to(s, t_now))
                    for s in S
                ])
            else:
                lat_before = float('inf')

            # new config
            lat_candidates = S + [v]
            lat_after = min([
                float(lat_map.get(c, {}).get(s, c.get_latency_to(s, t_now))) if lat_map is not None else float(c.get_latency_to(s, t_now))
                for s in lat_candidates
            ])

            curr_lats.append(lat_before if np.isfinite(lat_before) else 0.0)
            new_lats.append(lat_after if np.isfinite(lat_after) else 0.0)

        # average per-client min latency
        avg_before = float(np.mean(curr_lats)) if curr_lats else 0.0
        avg_after = float(np.mean(new_lats)) if new_lats else 0.0
        scenario_latency_deltas.append(avg_after - avg_before)

    gains = np.asarray(scenario_gains, dtype=float)
    # sanitize infinities and NaNs
    if gains.size == 0:
        return 0.0, {'mean_gain': 0.0, 'cvar': 0.0}
    gains = gains.copy()
    nan_mask = np.isnan(gains)
    gains[nan_mask] = 0.0
    pos_inf = np.isposinf(gains)
    neg_inf = np.isneginf(gains)
    LARGE = 1e12
    gains[pos_inf] = LARGE
    gains[neg_inf] = -LARGE

    mean_gain = float(np.mean(gains))
    std_gain = float(np.std(gains))

    # Compute CVaR of gains (lower tail = worst gains) with correct convention
    # For Eq. 15, we want α=0.95 for worst 5% tail
    cvar_gains = compute_cvar_empirical(gains, alpha=0.95)

    # Wasserstein DRO dual reformulation (Eq. 27):
    #   min_{λ≥0} { λε + (1/N) Σ_i sup_ξ [gain(ξ) - λ d(ξ, ξ_i)] }
    # With a log-SNR ground metric and gain as the objective, sup_ξ for scenario i
    # shifts each link's SNR to improve gain. We bound the per-scenario sup by the
    # Lipschitz sensitivity of gain to log-SNR, which is proportional to the
    # per-scenario gain spread. This yields a meaningful worst-case robust gain:
    #   robust ≈ λε + mean_i[gain_i] - λ * mean_i[sens_i]
    # We compute sens_i as the per-scenario absolute gain deviation (a local
    # sensitivity proxy), then minimize the dual over λ by bisection.
    sens = np.abs(gains - mean_gain)  # per-scenario sensitivity proxy
    mean_sens = float(np.mean(sens)) if sens.size else 0.0

    def dual_objective(lambda_dual):
        # sup_ξ [gain(ξ) - λ d(ξ, ξ_i)] ≈ gain_i + (sens_i - λ * sens_i).clip(min=0)
        # Simplification: gain_i - λ * sens_i (worst-case drop bounded by sensitivity)
        scenario_dual = gains - lambda_dual * sens
        return lambda_dual * epsilon + float(np.mean(scenario_dual))

    # Bisection to find optimal λ (minimizer of the dual, λ ≥ 0)
    lambda_opt = 0.0
    if mean_sens > 1e-12 and epsilon > 0:
        lo, hi = 0.0, max(1.0, (mean_gain - np.min(gains)) / max(mean_sens, 1e-12))
        best_val = float('inf')
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            val = dual_objective(mid)
            # Derivative of dual wrt λ: epsilon - mean_sens (constant here), so the
            # dual is linear in λ; optimum sits at a boundary. Track the best.
            if val < best_val:
                best_val = val
                lambda_opt = mid
            # Move toward the region that reduces the objective
            if epsilon - mean_sens > 0:
                hi = mid  # increasing λ raises objective → shrink
            else:
                lo = mid  # increasing λ lowers objective → grow
        # Evaluate endpoints explicitly since the dual is piecewise-linear
        for cand in (0.0, lo, hi, lambda_opt):
            val = dual_objective(cand)
            if val < best_val:
                best_val = val
                lambda_opt = cand

    # Robust gain via Wasserstein dual (worst-case expected gain)
    robust_gain = dual_objective(lambda_opt)

    # Blend with empirical CVaR so the tail risk directly shapes the score:
    # prefer the more conservative of the Wasserstein bound and the CVaR of gains.
    robust_gain = min(robust_gain, cvar_gains)

    # CVaR of latency deltas (upper tail for worst latency increases)
    latency_cvar = compute_cvar_empirical(scenario_latency_deltas, alpha=0.95) if len(scenario_latency_deltas) > 0 else 0.0

    # Combined robust objective: robust gain - latency penalty.
    # Skip the latency penalty on the very first pick (S empty): lat_before is
    # undefined and would otherwise penalize the best first server.
    if len(S) > 0:
        LAMBDA_LATENCY = 0.5
        robust_gain = robust_gain - LAMBDA_LATENCY * float(latency_cvar)

    # Fallback if robust gain is too small but mean gain is positive
    if robust_gain < 1e-6 and mean_gain > 0:
        robust_gain = mean_gain * 0.7

    return float(robust_gain), {
        'mean_gain': mean_gain,
        'cvar': float(cvar_gains),
        'latency_cvar': float(latency_cvar),
        'lambda_opt': lambda_opt,
        'epsilon': epsilon,
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
