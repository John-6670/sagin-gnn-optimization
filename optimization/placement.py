import logging

import numpy as np
from typing import List, Dict, Tuple, Callable

from simulation.topology.nodes import Node, NodeType
from optimization.objective import compute_objective, compute_marginal_gain, compute_placement_utility
from simulation.topology.aircomp import compute_amse_n, compute_amse_kn
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
                          target_snr=10.0, traffic=0.5, mobility=0.5, delta_list=None) -> Dict:
    """
    Improved OTA power control that actually adapts to channel quality.
    """
    return run_inner_ota_loop(
        selected_servers=selected_servers,
        clients=clients,
        t_now=t_now,
        target_snr=target_snr,
        traffic=traffic,
        mobility=mobility,
        inner_iterations=5,
        delta_list=delta_list
    )


def run_inner_ota_loop(
    selected_servers: List[Node],
    clients: List[Node],
    t_now,
    target_snr: float,
    traffic: float = 0.5,
    mobility: float = 0.5,
    inner_iterations: int = 5,
    beamformer_update_freq: int = 2,
    maml_lr: float = 0.01,
    delta_list=None
) -> Dict:
    """
    Two-timescale inner loop (Algorithm 3): Runs OTA adaptation at fast timescale
    given fixed placement S from outer loop.

    Alpha steps:
    1. Predictive sync via Kalman filter
    2. Per-tier adaptive beamforming (MRT for UAV/ground, zeroforce for sat)
    3. Power + pre-equalization phase control
    4. MAML meta-update at boundary (every T_slow / T_fast steps)

    Args:
        selected_servers: Placement set S from outer loop (DRO)
        clients: List of client nodes
        t_now: Current simulation time
        target_snr: Target SNR for power control
        traffic: Traffic load proxy
        mobility: Mobility proxy
        inner_iterations: Number of fast-timescale iterations (T_fast)
        beamformer_update_freq: Update beamformer every n iterations
        maml_lr: MAML inner learning rate

    Returns:
        dict: OTA control parameters per server per client
    """
    c = 3e8
    ota_results = {}
    power_saturations = 0

    # Initialize per-server RNG seed for reproducible sampling
    server_seeds = {s.id: hash(s.id) % 2**32 for s in selected_servers}

    for iter_idx in range(inner_iterations):
        for server in selected_servers:
            control_params = {}

            # === STEP 1: Predictive Sync via Kalman Filter ===
            # Get current channels and SNRs
            hs, snrs = [], []
            for client in clients:
                h = client.get_channel(server, t_now)
                snr_val = max(client.compute_snr_to(server, t_now), 1e-12)
                hs.append(np.array([h], dtype=np.complex128))
                snrs.append(snr_val)

            # MAML adaptation (runs at slow boundary, but called every iteration with small steps)
            adapted, sync_adj, _ = _MAML.maml_update(
                snrs, traffic=traffic, mobility=mobility, lr=maml_lr
            )
            power_scale = float(1 / (1 + np.exp(-adapted[0].item())))   # 0~1

            # === STEP 2: Per-tier Adaptive Beamforming ===
            # Update beamformer periodically (not every iteration to reduce overhead)
            if iter_idx % beamformer_update_freq == 0:
                if server.type == NodeType.SATELLITE:
                    # Satellite: zero-forcing beamforming to suppress inter-user interference
                    # Stack all client channels
                    H = np.vstack(hs)  # (num_clients, 1) - single antenna per client
                    if len(H) > 1:
                        # For single-antenna clients, MRT is optimal; ZF reduces to MRT
                        R = np.zeros((1, 1), dtype=np.complex128)
                        for h, snr in zip(hs, snrs):
                            R += snr * np.outer(h, h.conj())
                    else:
                        R = snrs[0] * np.outer(hs[0], hs[0].conj())
                else:
                    # UAV/Ground: MRT beamforming (maximizes array gain)
                    R = np.zeros((1, 1), dtype=np.complex128)
                    for h, snr in zip(hs, snrs):
                        R += snr * np.outer(h, h.conj())

                eigvals, eigvecs = np.linalg.eigh(R)
                w_star = eigvecs[:, np.argmax(eigvals)]
            else:
                # Reuse previous beamformer if available
                if iter_idx == 0:
                    # First iteration: compute initial beamformer
                    R = np.zeros((1, 1), dtype=np.complex128)
                    for h, snr in zip(hs, snrs):
                        R += snr * np.outer(h, h.conj())
                    eigvals, eigvecs = np.linalg.eigh(R)
                    w_star = eigvecs[:, np.argmax(eigvals)]

            # === STEP 3: Power Control + Pre-equalization Phase ===
            for idx, client in enumerate(clients):
                pos_c = client.get_position_at(t_now)
                pos_s = server.get_position_at(t_now)
                d_hat = np.linalg.norm(pos_c - pos_s) * 1000.0
                tau_base = d_hat / c

                # Kalman timing filter for predictive sync
                kf = KalmanTimingFilter(dt=1.0)
                kf.predict()
                tau_filtered = kf.update(tau_base)[0, 0]
                tau_window = tau_base + tau_filtered + sync_adj

                # Pre-equalization phase: compensate channel + add MAML phase + beamformer phase
                channel_phase = np.angle(hs[idx][0])
                pre_equalization_phase = -channel_phase + float(adapted[1].item()) + np.angle(w_star[0])

                # Energy-minimizing power calculation: achieve target fraction of max SNR
                # MAML learns to minimize power while maintaining target_snr
                max_possible_snr = snrs[idx]
                base_required_power = min(client.power, target_snr * client.power / max(max_possible_snr, 1e-6))
                optimized_power = base_required_power * power_scale

                # Clip power: [noise_floor, max_client_power]
                pre_clipped = optimized_power
                optimized_power = float(np.clip(optimized_power, 1e-4, client.power))

                if optimized_power >= client.power - 1e-6:
                    power_saturations += 1

                # Compute AMSE for this client-server pair as monitoring metric
                if iter_idx == inner_iterations - 1:  # Only on last iteration
                    amse_val = compute_amse_kn(client, server, delta_list, t_now)
                else:
                    amse_val = 0.0

                control_params[client.id] = {
                    "power": optimized_power,
                    "phase": pre_equalization_phase,
                    "snr": snrs[idx],
                    "tau_window": float(tau_window),
                    "beamformer": w_star.copy() if iter_idx == inner_iterations - 1 else None,
                    "requested_power": float(pre_clipped),
                    "max_power_clipped": optimized_power >= client.power - 1e-6,
                    "amse_kn": float(amse_val),
                    "iter": iter_idx,
                }

            ota_results[server.id] = control_params

    if power_saturations > 0:
        log.debug(
            "run_inner_ota_loop: %d/%d client-server pairs saturated to max power (%d iters)",
            power_saturations, len(clients) * len(selected_servers), inner_iterations
        )

    # Final MAML meta-update at slow boundary (called once per outer step)
    # Use SNRs from the last iteration of the last server
    if selected_servers and clients:
        last_snrs = []
        last_server = selected_servers[-1]
        for client in clients:
            snr_val = max(client.compute_snr_to(last_server, t_now), 1e-12)
            last_snrs.append(snr_val)
        _MAML.maml_update(last_snrs, traffic=traffic, mobility=mobility, lr=maml_lr, meta_update=True)

    return ota_results


def greedy_server_selection(candidates, clients, budget, cost: dict[Node, float], thresh, alpha, beta, delta_list, N,
                            t_now=None, target_snr=1.0, energy_weight=0.3, use_hierarchical=True) -> List[Node]:
    # Build SNR map - use orbital-average SNR for satellite links per Eq. 21
    from optimization.objective import compute_orbital_avg_snr
    from simulation.topology.nodes import NodeType

    snr_map = {c: {} for c in clients}
    for c in clients:
        for s in candidates:
            if s.type == NodeType.SATELLITE:
                snr_map[c][s] = compute_orbital_avg_snr(c, s, t_now)
            else:
                snr_map[c][s] = max(c.compute_snr_to(s, t_now), 1e-12)
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
        # Use hierarchical AMSE for objective computation
        current_utility = compute_objective(S, clients, alpha, beta, delta_list, snr_map=snr_map, use_hierarchical=use_hierarchical, t_now=t_now)

        for server in candidates_cp:
            new_utility = compute_objective(S + [server], clients, alpha, beta, delta_list, snr_map=snr_map, use_hierarchical=use_hierarchical, t_now=t_now)
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
                                gnn_checkpoint=None, kappa=0.3, tau_amse=None):
    from optimization.dro import bisect_lambda_for_amse_target
    # Increase kappa from 0.15 to 0.30 to match greedy's pruning level (30% instead of 15%)
    C = _gnn_prune_candidates(candidates, clients, kappa=kappa, checkpoint=gnn_checkpoint)
    # C = candidates.copy()
    for c in C:
        print(f"Candidate {c.id} | Type: {c.type} | Cost: {cost[c]:.2f}")
    # Use orbital-average SNR for outer loop placement decisions (Eq. 21)
    scenario_maps = sample_snr_scenarios(clients, C, t_now, N=N,
                                        coherence_time=coherence_time, sigma_snr=sigma_snr,
                                        use_orbital_avg=True)
    # Precompute latency map for this selection time to avoid repeated expensive calls
    print(f"DR-Greedy: Sampled {N} SNR scenarios for {len(clients)} clients and {len(C)} candidates")
    latency_map = {c: {s: c.get_latency_to(s, t_now) for s in C} for c in clients}
    all_latencies = [latency_map[c][s] for c in clients for s in C]
    import numpy as _np
    latency_scale = max(_np.percentile(all_latencies, 95) if all_latencies else 1.0, 1e-6)

    amse_scale_values = []
    if clients:
        sigma2 = float(np.mean([c.noise_variance for c in clients]))
        gradient_dim = clients[0].gradient_dim
        from simulation.topology.aircomp import compute_amse_hierarchical
        tier_sync_errors = {1: 1e-9, 2: 5e-9, 3: 1e-8}
        for snr_map in scenario_maps:
            # Use hierarchical AMSE for scaling, honoring the scenario snr_map
            amse_val = compute_amse_hierarchical(C, clients, delta_list, t_now=t_now, tier_sync_errors=tier_sync_errors, snr_map=snr_map)
            amse_scale_values.append(amse_val)
    amse_scale = max(_np.percentile(amse_scale_values, 95) if amse_scale_values else 1.0, 1e-12)

    bundle = ScenarioBundle(scenarios=scenario_maps, clients=clients, candidates=C,
                            delta_list=delta_list, alpha=alpha, beta=beta,
                            latency_map=latency_map, latency_scale=latency_scale,
                            amse_scale=amse_scale, t_now=t_now, use_hierarchical=True)
    print(f"DR-Greedy: Starting robust selection with {len(C)} candidates and budget {budget}")

    S, total_cost, remaining = [], 0.0, list(C)
    while remaining:
        best_v, best_score = None, -float('inf')
        for v in remaining:
            if total_cost + cost[v] > budget:
                continue

            score, _ = robust_marginal_gain(S, v, bundle, epsilon=epsilon, alpha_cvar=alpha_cvar)
            # Normalize by cost: a 10-cost satellite must deliver 10x the robust
            # gain of a 1-cost ground node to be worth half the budget. Without
            # this, DR over-buys expensive satellites on raw (absolute) gain.
            score = score / max(cost[v], 1e-9)
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
        empty_obj = compute_objective([], clients, alpha, beta, delta_list, use_hierarchical=True, t_now=t_now)
        best_v = None
        best_val = float('inf')
        for v in C:
            if cost[v] > budget:
                continue
            value = compute_objective(
                [v], clients, alpha, beta, delta_list, latency_map=latency_map,
                latency_scale=latency_scale, amse_scale=amse_scale,
                use_hierarchical=True, t_now=t_now,
            )
            if value < best_val:
                best_val = value
                best_v = v

        if best_v is not None and best_val < empty_obj:
            S.append(best_v)

    return local_one_swap(S, C, budget, cost, bundle, epsilon=epsilon, alpha_cvar=alpha_cvar)
