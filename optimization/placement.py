import numpy as np
from typing import List, Dict

from simulation.topology.nodes import Node, NodeType
from optimization.objective import compute_utility
from optimization.dro import ScenarioBundle, sample_snr_scenarios, robust_marginal_gain, local_one_swap
from optimization.meta_learner import MAMLInnerOptimizer
from models.gnn.train import predict_candidate_scores


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


def predictive_ota_control(selected_servers: List[Node], clients: List[Node], t_now, target_snr=10.0, traffic=0.5, mobility=0.5) -> Dict:
    ota_results = {}
    c = 3e8
    
    for server in selected_servers:
        control_params = {}
        
        kf = KalmanTimingFilter(dt=1.0)
        hs, snrs = [], []
        
        for client in clients:
            h = client.get_channel(server, t_now)
            snr_val = client.compute_snr_to(server, t_now)
            hs.append(np.array([h], dtype=np.complex128))
            snrs.append(max(snr_val, 1e-12))
            
        # Per-tier MRT-like beamforming (rank-1 channels in current simulator)
        R = np.zeros((1, 1), dtype=np.complex128)
        for h, snr in zip(hs, snrs):
            R += snr * np.outer(h, h.conj())
        eigvals, eigvecs = np.linalg.eigh(R)
        w_star = eigvecs[:, np.argmax(eigvals)]

        adapted, sync_adj, _ = _MAML.maml_update(snrs, traffic=traffic, mobility=mobility)
        power_scale = float(1 / (1 + np.exp(-adapted[0].item())))
        phase_bias = float(adapted[1].item())

        for idx, client in enumerate(clients):
            pos_c = client.get_position_at(t_now)
            pos_s = server.get_position_at(t_now)
            d_hat = np.linalg.norm(pos_c - pos_s) * 1000.0
            tau_base = d_hat / c
            kf.predict()
            tau_filtered = kf.update(tau_base)[0, 0]
            tau_window = tau_base + tau_filtered + sync_adj

            channel_phase = np.angle(hs[idx][0])
            pre_equalization_phase = -channel_phase + phase_bias + np.angle(w_star[0])
            required_power = target_snr / max(snrs[idx], 1e-9)

            channel_penalty = 1.0 + abs(channel_phase) / np.pi
            mobility_penalty = 1.0 + mobility
            traffic_penalty = 1.0 + traffic

            optimized_power = required_power
            optimized_power *= channel_penalty
            optimized_power *= mobility_penalty
            optimized_power *= traffic_penalty
            optimized_power *= power_scale

            optimized_power = float(np.clip(
                optimized_power,
                1e-4,
                client.power,
            ))
            
            control_params[client.id] = {
                "power": optimized_power,
                "phase": pre_equalization_phase,
                "snr": snrs[idx],
                "tau_window": float(tau_window),
                "beamformer": w_star.copy(),
            }
            
        ota_results[server.id] = control_params
        
    return ota_results


def greedy_server_selection(candidates, clients, budget, cost: dict[Node, float], thresh, alpha, beta, delta_list, N, t_now=None) -> List[Node]:
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

        sorted_candidates = sorted(
            candidates_cp,
            key=lambda x: scores.get(x, -1e9),
            reverse=True
        )

        top_k = max(5, int(0.3 * len(sorted_candidates)))
        candidates_cp = sorted_candidates[:top_k]

    while candidates_cp:
        best_server, best_gain = None, -float("inf")
        current_utility = compute_utility(S, clients, alpha, beta, delta_list, snr_map=snr_map)

        for server in candidates_cp:
            new_utility = compute_utility(S + [server], clients, alpha, beta, delta_list, snr_map=snr_map)

            gain = (current_utility - new_utility) / cost[server]
            if gain > best_gain:
                best_gain, best_server = gain, server

        if best_server is None:
            break

        delta_snr = sum(max(0.0, snr_map[c][best_server] - best_snr[c]) for c in clients)
        if delta_snr > thresh and total_cost + cost[best_server] <= budget:
            S.append(best_server)
            total_cost += cost[best_server]
            # Update best_snr for the next iteration
            for c in clients:
                best_snr[c] = max(best_snr[c], snr_map[c][best_server])
            if total_cost >= budget:
                break

        candidates_cp.remove(best_server)
        
    return S


def _gnn_prune_candidates(candidates, clients, kappa=0.15, checkpoint=None):
    try:
        from models.gnn.train import load_gnn_for_inference, predict_candidate_scores
        model = load_gnn_for_inference(checkpoint)
        scored = predict_candidate_scores(model, candidates, clients)
        keep = max(1, int(np.ceil(kappa * len(candidates))))
        return [c for c, _ in sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:keep]]
    except Exception:
        return candidates


def dr_greedy_server_selection(candidates, clients, budget, cost, thresh, alpha, beta, delta_list, t_now=None,
                                epsilon=0.1, alpha_cvar=0.9, N=64, coherence_time=25.0, sigma_snr=0.35,
                                gnn_checkpoint=None, kappa=0.30, tau_amse=None):
    from optimization.dro import bisect_lambda_for_amse_target
    # Increase kappa from 0.15 to 0.30 to match greedy's pruning level (30% instead of 15%)
    C = _gnn_prune_candidates(candidates, clients, kappa=kappa, checkpoint=gnn_checkpoint)
    scenario_maps = sample_snr_scenarios(clients, C, t_now, N=N,
                                        coherence_time=coherence_time, sigma_snr=sigma_snr)
    # Precompute latency map for this selection time to avoid repeated expensive calls
    latency_map = {c: {s: c.get_latency_to(s, t_now) for s in C} for c in clients}
    bundle = ScenarioBundle(scenarios=scenario_maps, clients=clients, candidates=C,
                            delta_list=delta_list, alpha=alpha, beta=beta, latency_map=latency_map)
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
        if best_v is None or best_score < -1e-6:
            break
        S.append(best_v)
        total_cost += cost[best_v]
        remaining.remove(best_v)
        # Algorithm 2 Phase 2: enforce CVaR AMSE target (Eq. 37)
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
