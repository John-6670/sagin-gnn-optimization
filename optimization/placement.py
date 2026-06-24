import logging

import numpy as np
from typing import List, Dict

from simulation.topology.nodes import Node, NodeType
from optimization.objective import compute_utility
from optimization.dro import ScenarioBundle, sample_snr_scenarios, robust_marginal_gain, local_one_swap
from optimization.meta_learner import MAMLInnerOptimizer
from models.gnn.train import predict_candidate_scores

log = logging.getLogger(__name__)


def _get_tier_budgets(budget, num_sats=36, num_uavs=8, num_ground=20):
    return {
        NodeType.SATELLITE: 1,
        NodeType.UAV: max(2, int(budget * 0.4)),
        NodeType.GROUND: max(5, int(budget * 0.3)),
    }


class KalmanTimingFilter:
    def __init__(self, dt=1.0, q_var=1e-10, r_var=1e-9):
        self.dt = dt
        self.x = np.zeros((2, 1))
        self.P = np.eye(2)
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = q_var * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        self.R = np.array([[r_var]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z_measured):
        z = np.array([[z_measured]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x


_MAML = MAMLInnerOptimizer()


def predictive_ota_control(selected_servers: List[Node], clients: List[Node], t_now, 
                          target_snr=10.0, traffic=0.5, mobility=0.5) -> Dict:
    """
    Improved OTA power control that actually adapts to channel quality.
    """
    ota_results = {}
    c = 3e8
    power_saturations = 0

    for server in selected_servers:
        control_params = {}
        
        # Get current channels and SNRs
        hs, snrs = [], []
        for client in clients:
            h = client.get_channel(server, t_now)
            snr_val = max(client.compute_snr_to(server, t_now), 1e-12)
            hs.append(np.array([h], dtype=np.complex128))
            snrs.append(snr_val)

        # Simple MRT beamforming
        R = np.zeros((1, 1), dtype=np.complex128)
        for h, snr in zip(hs, snrs):
            R += snr * np.outer(h, h.conj())
        eigvals, eigvecs = np.linalg.eigh(R)
        w_star = eigvecs[:, np.argmax(eigvals)]

        # MAML adaptation
        adapted, sync_adj, _ = _MAML.maml_update(snrs, traffic=traffic, mobility=mobility)
        power_scale = float(1 / (1 + np.exp(-adapted[0].item())))   # 0~1

        for idx, client in enumerate(clients):
            pos_c = client.get_position_at(t_now)
            pos_s = server.get_position_at(t_now)
            d_hat = np.linalg.norm(pos_c - pos_s) * 1000.0
            tau_base = d_hat / c

            kf = KalmanTimingFilter(dt=1.0)  # Note: better to reuse KF per server if possible
            kf.predict()
            tau_filtered = kf.update(tau_base)[0, 0]
            tau_window = tau_base + tau_filtered + sync_adj

            channel_phase = np.angle(hs[idx][0])
            pre_equalization_phase = -channel_phase + float(adapted[1].item()) + np.angle(w_star[0])

            # === ENERGY-MINIMIZING POWER CALCULATION ===
            # Target 80% of max achievable SNR — always feasible regardless of channel.
            # MAML is trained to minimize power further while maintaining this fraction.
            base_required_power = 0.8 * client.power
            optimized_power = base_required_power * power_scale

            # Clip — never exceed client.power, never go below noise floor
            pre_clipped = optimized_power
            optimized_power = float(np.clip(optimized_power, 1e-4, client.power))

            if optimized_power >= client.power - 1e-6:
                power_saturations += 1

            control_params[client.id] = {
                "power": optimized_power,
                "phase": pre_equalization_phase,
                "snr": snrs[idx],
                "tau_window": float(tau_window),
                "beamformer": w_star.copy(),
                "requested_power": float(pre_clipped),
                "max_power_clipped": optimized_power >= client.power - 1e-6,
            }

        ota_results[server.id] = control_params

    if power_saturations > 0:
        log.debug(
            "predictive_ota_control: %d/%d client-server pairs saturated to max power",
            power_saturations, len(clients) * len(selected_servers)
        )

    return ota_results


def greedy_server_selection(candidates, clients, budget, cost: dict[Node, float], thresh, alpha, beta, delta_list, N,
                            t_now=None, target_snr=1.0, energy_weight=0.3) -> List[Node]:
    snr_map = {c: {s: c.compute_snr_to(s, t_now) for s in candidates} for c in clients}
    best_snr = {c: 0.0 for c in clients}
    S, total_cost = [], 0
    candidates_cp = candidates.copy()

    if hasattr(greedy_server_selection, 'gnn_model'):
        scores = predict_candidate_scores(
            greedy_server_selection.gnn_model,
            candidates_cp,
            clients,
            selected_servers=[]
        )
        sorted_candidates = sorted(candidates_cp, key=lambda x: scores.get(x, -1e9), reverse=True)
        top_k = min(len(sorted_candidates), max(10, int(0.5 * len(sorted_candidates))))
        candidates_cp = sorted_candidates[:top_k]

    while candidates_cp:
        # Early stop: all clients already meet target_snr from some selected server
        if S and all(best_snr[c] >= target_snr for c in clients):
            break

        best_server, best_gain = None, -float("inf")
        current_utility = compute_utility(S, clients, alpha, beta, delta_list, snr_map=snr_map)

        for server in candidates_cp:
            new_utility = compute_utility(S + [server], clients, alpha, beta, delta_list, snr_map=snr_map)
            utility_gain = (current_utility - new_utility) / cost[server]

            # Energy proxy: fraction of clients whose SNR target this server satisfies.
            # High value → clients can transmit at lower power → less energy.
            energy_proxy = sum(
                min(snr_map[c][server] / max(target_snr, 1e-9), 1.0) for c in clients
            ) / max(len(clients), 1)

            gain = utility_gain + energy_weight * energy_proxy
            if gain > best_gain:
                best_gain, best_server = gain, server

        if best_server is None:
            break

        delta_snr = sum(max(0.0, snr_map[c][best_server] - best_snr[c]) for c in clients)
        if delta_snr > thresh and total_cost + cost[best_server] <= budget:
            S.append(best_server)
            total_cost += cost[best_server]
            for c in clients:
                best_snr[c] = max(best_snr[c], snr_map[c][best_server])
            if total_cost >= budget:
                break

        candidates_cp.remove(best_server)

    return S


def _gnn_prune_candidates(candidates, clients, kappa=0.3, checkpoint=None):
    try:
        from models.gnn.train import load_gnn_for_inference, predict_candidate_scores
        import numpy as np
        from collections import defaultdict
        
        model = load_gnn_for_inference(checkpoint)
        scored = predict_candidate_scores(model, candidates, clients)
        
        grouped = defaultdict(list)
        for c in candidates:
            grouped[c.type].append(c)
            
        final_candidates = []
        for c_type, group in grouped.items():
            keep = max(1, int(np.ceil(kappa * len(group))))
            sorted_group = sorted(group, key=lambda x: scored.get(x, -1e9), reverse=True)
            final_candidates.extend(sorted_group[:keep])
            
        return final_candidates
    except Exception as e:
        import logging
        logging.warning(f"GNN pruning failed: {e}")
        return candidates


def dr_greedy_server_selection(candidates, clients, budget, cost, thresh, alpha, beta, delta_list, t_now=None,
                                epsilon=0.15, alpha_cvar=0.10, N=64, coherence_time=25.0, sigma_snr=0.35,
                                gnn_checkpoint=None, kappa=0.5, tau_amse=None):
    from optimization.dro import bisect_lambda_for_amse_target
    # Increase kappa from 0.15 to 0.30 to match greedy's pruning level (30% instead of 15%)
    C = _gnn_prune_candidates(candidates, clients, kappa=kappa, checkpoint=gnn_checkpoint)
    # C = candidates.copy()
    for c in C:
        print(f"Candidate {c.id} | Type: {c.type} | Cost: {cost[c]:.2f}")
    scenario_maps = sample_snr_scenarios(clients, C, t_now, N=N,
                                        coherence_time=coherence_time, sigma_snr=sigma_snr)
    # Precompute latency map for this selection time to avoid repeated expensive calls
    print(f"DR-Greedy: Sampled {N} SNR scenarios for {len(clients)} clients and {len(C)} candidates")
    latency_map = {c: {s: c.get_latency_to(s, t_now) for s in C} for c in clients}
    bundle = ScenarioBundle(scenarios=scenario_maps, clients=clients, candidates=C,
                            delta_list=delta_list, alpha=alpha, beta=beta, latency_map=latency_map)
    print(f"DR-Greedy: Starting robust selection with {len(C)} candidates and budget {budget}")
    
    S, total_cost, remaining = [], 0.0, list(C)   
    while remaining:
        best_v, best_score = None, -float('inf')
        for v in remaining:
            if total_cost + cost[v] > budget:
                continue
            
            score, _ = robust_marginal_gain(S, v, bundle, epsilon=epsilon, alpha_cvar=alpha_cvar)
            if score > best_score:
                best_score, best_v = score, v
        # Stop only when no feasible server exists or gain is genuinely negative
        if best_v is None:
            break
        if best_score < -1 and len(S) > 0:
            break
        S.append(best_v)
        total_cost += cost[best_v]
        remaining.remove(best_v)

        if tau_amse is not None:
            bisect_lambda_for_amse_target(S, bundle, alpha_cvar, tau_amse)

    if not S:
        empty_obj = compute_utility([], clients, alpha, beta, delta_list)
        best_v = None
        best_val = float('inf')
        for v in C:
            if cost[v] > budget:
                continue
            value = compute_utility([v], clients, alpha, beta, delta_list, latency_map=latency_map)
            if value < best_val:
                best_val = value
                best_v = v

        if best_v is not None and best_val < empty_obj:
            S.append(best_v)

    return local_one_swap(S, C, budget, cost, bundle, epsilon=epsilon, alpha_cvar=alpha_cvar)
