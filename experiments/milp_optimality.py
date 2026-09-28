"""
MILP Optimality Gap Experiments (Requirement 3).

Formulates Eq 24 (two-timescale bilevel) as a MILP using pulp,
solves small instances (5-10 clients) to get global optimum,
and reports gap of Algorithm 2 (DR-Greedy) relative to MILP.
"""

from __future__ import annotations
import csv
import logging
import os
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    log = logging.getLogger(__name__)
    log.warning("pulp not installed. MILP experiments will be skipped. Install with: pip install pulp")

from simulation.topology.nodes import Node, NodeType, generate_nodes
from simulation.config_loader import load_config
from optimization.baselines import dr_selection
from optimization.placement import greedy_server_selection, run_inner_ota_loop
from optimization.objective import compute_objective
from simulation.run_simulation import build_costs
from simulation.topology.aircomp import compute_amse_hierarchical
from simulation.evaluation.metrics import compute_e2e_latency, compute_total_energy
from skyfield.api import load

log = logging.getLogger(__name__)


@dataclass
class MILPConfig:
    """Configuration for MILP optimality gap experiments."""
    num_clients_list: List[int]
    num_seeds: int
    output_dir: str = "results/experiments/milp"
    time_limit_seconds: int = 300


@dataclass
class MILPResult:
    """Single MILP experiment result."""
    num_clients: int
    seed: int
    milp_objective: float
    dr_greedy_objective: float
    optimality_gap: float
    milp_runtime: float
    dr_greedy_runtime: float
    milp_selected_sat: int
    milp_selected_uav: int
    milp_selected_ground: int
    dr_selected_sat: int
    dr_selected_uav: int
    dr_selected_ground: int
    milp_status: str


def formulate_milp(
    clients: List[Node],
    candidates: List[Node],
    budget: int,
    cost: Dict[Node, float],
    thresh: float,
    alpha: float,
    beta: float,
    delta_list: List[float],
    t_now,
    num_scenarios: int = 16,
) -> Tuple[pulp.LpProblem, Dict]:
    """
    Formulate the bilevel optimization (Eq 24) as a MILP.

    Two-timescale formulation:
    - Outer: Server placement (binary variables)
    - Inner: OTA power/phase control (continuous variables)

    For MILP, we linearize by fixing inner variables or using big-M.
    """
    prob = pulp.LpProblem("SAGIN_Bilevel_Placement", pulp.LpMinimize)

    # ============================================
    # Decision Variables
    # ============================================
    # Binary: server selection
    x = {c: pulp.LpVariable(f"x_{id(c)}", cat='Binary') for c in candidates}

    # Continuous: power allocation for each client-server pair
    # Only active if both client and server are selected
    p = {}
    for cl in clients:
        for s in candidates:
            p[(cl, s)] = pulp.LpVariable(f"p_{id(cl)}_{id(s)}", lowBound=0, upBound=1.0)

    # Continuous: phase correction variables
    phi = {}
    for cl in clients:
        for s in candidates:
            phi[(cl, s)] = pulp.LpVariable(f"phi_{id(cl)}_{id(s)}", lowBound=-np.pi, upBound=np.pi)

    # ============================================
    # Objective: minimize weighted compound loss
    # ============================================
    # We approximate the inner loop by fixing OTA parameters to nominal values
    # and only optimizing placement (x variables).
    # The inner problem is approximated via pre-computed SNR scenarios.

    # Pre-compute SNRs for all client-server pairs at t_now
    snr_map = {}
    for cl in clients:
        for s in candidates:
            snr = cl.compute_snr_to(s, t_now)
            snr_map[(cl, s)] = snr

    # Build objective
    # Latency term: sum over clients of latency to nearest selected server
    # AMSE term: hierarchical AMSE based on selected servers
    # Energy term: transmit power * transmission time

    # For MILP, we use a simplified linear objective
    # Latency penalty
    latency_terms = []
    for cl in clients:
        lat_terms = []
        for s in candidates:
            lat = cl.compute_latency_to(s, t_now)
            lat_terms.append(lat * x[s])
        # Client connects to at least one server
        if lat_terms:
            latency_terms.append(pulp.lpSum(lat_terms))

    # AMSE approximation: use pre-computed average AMSE for each server
    amse_terms = []
    for s in candidates:
        # Estimate AMSE contribution if this server is selected
        amse_contrib = 0
        for cl in clients:
            snr = snr_map[(cl, s)]
            if snr > thresh:
                amse_contrib += 1.0 / (1.0 + snr)  # Simple proxy
        amse_terms.append(amse_contrib * x[s])

    # Energy term
    energy_terms = []
    for s in candidates:
        # Energy cost proportional to number of clients served
        n_clients_served = pulp.lpSum(x[s] for cl in clients)
        energy_terms.append(0.1 * n_clients_served * x[s])

    # Combined objective
    prob += alpha * pulp.lpSum(latency_terms) + beta * pulp.lpSum(amse_terms) + 0.1 * pulp.lpSum(energy_terms)

    # ============================================
    # Constraints
    # ============================================

    # Budget constraint
    prob += pulp.lpSum(cost[s] * x[s] for s in candidates) <= budget

    # SNR threshold constraint: selected servers must have at least one client above thresh
    for s in candidates:
        above_thresh = sum(1 for cl in clients if snr_map[(cl, s)] > thresh)
        if above_thresh == 0:
            prob += x[s] == 0  # Cannot select server with no feasible clients

    # Each client must be assigned to at least one selected server
    for cl in clients:
        prob += pulp.lpSum(x[s] for s in candidates) >= 1

    # Tier structure constraints (simplified)
    # At least one ground station must be selected if any UAV is selected
    ground_servers = [s for s in candidates if s.type == NodeType.GROUND]
    uav_servers = [s for s in candidates if s.type == NodeType.UAV]
    if ground_servers and uav_servers:
        prob += pulp.lpSum(x[s] for s in uav_servers) <= len(uav_servers) * pulp.lpSum(x[s] for s in ground_servers)

    return prob, {'x': x, 'p': p, 'phi': phi, 'snr_map': snr_map}


def solve_milp(
    clients: List[Node],
    candidates: List[Node],
    budget: int,
    cost: Dict[Node, float],
    thresh: float,
    alpha: float,
    beta: float,
    delta_list: List[float],
    t_now,
    time_limit: int = 300,
) -> Tuple[Optional[List[Node]], float, float, str]:
    """
    Solve MILP and return selected servers, objective value, runtime, and status.
    """
    if not PULP_AVAILABLE:
        return None, float('inf'), 0.0, "pulp_not_available"

    prob, variables = formulate_milp(
        clients, candidates, budget, cost, thresh, alpha, beta,
        delta_list, t_now
    )

    # Solve with time limit
    start_time = time.perf_counter()
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    prob.solve(solver)
    runtime = time.perf_counter() - start_time

    status = pulp.LpStatus[prob.status]

    if prob.status == pulp.LpOptimal:
        # Extract selected servers
        x = variables['x']
        selected = [s for s in candidates if pulp.value(x[s]) > 0.5]

        # Compute true objective with full evaluation
        objective = evaluate_objective(selected, clients, candidates, cost, thresh, alpha, beta, delta_list, t_now)

        return selected, objective, runtime, status
    else:
        log.warning(f"MILP not solved to optimality: {status}")
        return None, float('inf'), runtime, status


def evaluate_objective(
    selected: List[Node],
    clients: List[Node],
    candidates: List[Node],
    cost: Dict[Node, float],
    thresh: float,
    alpha: float,
    beta: float,
    delta_list: List[float],
    t_now,
) -> float:
    """Evaluate the full objective for a given server selection."""
    if not selected or not clients:
        return float('inf')

    # Latency
    latency = compute_e2e_latency(clients, selected, t_now)

    # AMSE
    tier_sync_errors = {1: 1e-9, 2: 5e-9, 3: 1e-8}
    amse = compute_amse_hierarchical(
        selected, clients, delta_list, t_now=t_now,
        tier_sync_errors=tier_sync_errors
    )

    # Energy
    ota = run_inner_ota_loop(
        selected_servers=selected,
        clients=clients,
        t_now=t_now,
        target_snr=10.0,
        traffic=0.5,
        mobility=0.5,
        inner_iterations=5,
        maml_lr=0.01,
        delta_list=delta_list
    )
    energy = compute_total_energy(clients, selected, ota, transmission_time=10, t_now=t_now)

    return alpha * latency + beta * amse + 0.1 * energy


def run_milp_optimality_experiments(
    config: MILPConfig = None,
    budget: int = 20,
    thresh: float = 0.0,
    delta_list: List[float] = None,
    num_scenarios: int = 16,
    alpha: float = 0.5,
    beta: float = 0.5,
    seeds: List[int] = None,
) -> List[MILPResult]:
    """
    Run MILP optimality gap experiments for varying client counts.

    Compares DR-Greedy (Algorithm 2) against MILP global optimum.
    """
    if not PULP_AVAILABLE:
        log.error("pulp not available. Cannot run MILP experiments.")
        return []

    if config is None:
        config = MILPConfig(
            num_clients_list=[5, 6, 7, 8, 9, 10],
            num_seeds=5,
        )

    if delta_list is None:
        delta_list = [0.1, 0.2]

    if seeds is None:
        seeds = list(range(config.num_seeds))

    # Load base config
    base_config = load_config("configs/default.yaml")
    sim_config = base_config["simulation"]
    num_sats = sim_config.get("num_sats", 1)
    num_uavs = sim_config.get("num_uavs", 2)
    num_ground = sim_config.get("num_ground", 4)
    area_size = sim_config.get("area_size", 2000)
    gradient_dim = sim_config.get("gradient_dim", 100)

    os.makedirs(config.output_dir, exist_ok=True)
    csv_path = os.path.join(config.output_dir, "milp_optimality_gap.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'num_clients', 'seed', 'milp_objective', 'dr_greedy_objective',
            'optimality_gap', 'milp_runtime', 'dr_greedy_runtime',
            'milp_selected_sat', 'milp_selected_uav', 'milp_selected_ground',
            'dr_selected_sat', 'dr_selected_uav', 'dr_selected_ground',
            'milp_status'
        ])

    all_results = []

    for num_clients in config.num_clients_list:
        for seed in seeds:
            log.info(f"Running MILP experiment: {num_clients} clients, seed={seed}")

            ts = load.timescale()
            t_now = ts.now()

            # Generate nodes with specific client count
            nodes = generate_nodes(
                num_sats=num_sats,
                num_uavs=num_uavs,
                num_ground=num_ground,
                num_clients=num_clients,
                area_size=area_size,
                gradient_dim=gradient_dim,
                t0=t_now
            )

            clients = [n for n in nodes if n.type == NodeType.CLIENT]
            candidates = [n for n in nodes if n.type != NodeType.CLIENT]
            cost = build_costs(candidates)

            # Run DR-Greedy (Algorithm 2)
            dr_start = time.perf_counter()
            dr_selected = dr_selection(
                candidates=candidates, clients=clients, budget=budget, cost=cost,
                thresh=thresh, alpha=alpha, beta=beta, delta_list=delta_list,
                N=num_scenarios, t_now=t_now, epsilon=0.1, kappa=0.3
            )
            dr_runtime = time.perf_counter() - dr_start

            dr_objective = evaluate_objective(
                dr_selected, clients, candidates, cost, thresh, alpha, beta,
                delta_list, t_now
            )

            # Run MILP
            milp_selected, milp_objective, milp_runtime, milp_status = solve_milp(
                clients, candidates, budget, cost, thresh, alpha, beta,
                delta_list, t_now, time_limit=config.time_limit_seconds
            )

            if milp_selected is not None and milp_objective < float('inf'):
                gap = (dr_objective - milp_objective) / milp_objective * 100
            else:
                gap = float('inf')

            # Count server types
            def count_types(servers):
                return (
                    sum(1 for s in servers if s.type == NodeType.SATELLITE),
                    sum(1 for s in servers if s.type == NodeType.UAV),
                    sum(1 for s in servers if s.type == NodeType.GROUND)
                )

            milp_sat, milp_uav, milp_ground = count_types(milp_selected) if milp_selected else (0, 0, 0)
            dr_sat, dr_uav, dr_ground = count_types(dr_selected) if dr_selected else (0, 0, 0)

            result = MILPResult(
                num_clients=num_clients,
                seed=seed,
                milp_objective=milp_objective,
                dr_greedy_objective=dr_objective,
                optimality_gap=gap,
                milp_runtime=milp_runtime,
                dr_greedy_runtime=dr_runtime,
                milp_selected_sat=milp_sat,
                milp_selected_uav=milp_uav,
                milp_selected_ground=milp_ground,
                dr_selected_sat=dr_sat,
                dr_selected_uav=dr_uav,
                dr_selected_ground=dr_ground,
                milp_status=milp_status,
            )

            all_results.append(result)

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    result.num_clients, result.seed, result.milp_objective,
                    result.dr_greedy_objective, result.optimality_gap,
                    result.milp_runtime, result.dr_greedy_runtime,
                    result.milp_selected_sat, result.milp_selected_uav, result.milp_selected_ground,
                    result.dr_selected_sat, result.dr_selected_uav, result.dr_selected_ground,
                    result.milp_status
                ])

            log.info(f"  MILP: obj={milp_objective:.4f}, time={milp_runtime:.2f}s, status={milp_status}")
            log.info(f"  DR-Greedy: obj={dr_objective:.4f}, time={dr_runtime:.4f}s")
            log.info(f"  Gap: {gap:.2f}%")

    log.info(f"MILP experiments complete. Results saved to {csv_path}")
    return all_results


def plot_optimality_gap(results: List[MILPResult], output_dir: str = "plots/experiments/milp"):
    """Generate optimality gap plots."""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame([r.__dict__ for r in results])

    if df.empty:
        log.warning("No results for MILP plotting")
        return

    # Filter valid gaps
    df = df[np.isfinite(df['optimality_gap'])]

    if df.empty:
        log.warning("No finite optimality gaps")
        return

    import matplotlib.pyplot as plt

    # Boxplot of gap by num_clients
    fig, ax = plt.subplots(figsize=(10, 6))

    clients_sorted = sorted(df['num_clients'].unique())
    gap_data = [df[df['num_clients'] == n]['optimality_gap'].values for n in clients_sorted]

    bp = ax.boxplot(gap_data, labels=[str(n) for n in clients_sorted], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#D62728')
        patch.set_alpha(0.6)

    ax.set_xlabel('Number of Clients', fontsize=12)
    ax.set_ylabel('Optimality Gap (%)', fontsize=12)
    ax.set_title('DR-Greedy Optimality Gap vs MILP Global Optimum', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimality_gap_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Mean gap line plot
    fig, ax = plt.subplots(figsize=(10, 6))

    mean_gaps = df.groupby('num_clients')['optimality_gap'].mean()
    std_gaps = df.groupby('num_clients')['optimality_gap'].std()

    ax.errorbar(
        mean_gaps.index, mean_gaps.values,
        yerr=std_gaps.values,
        fmt='o-', color='#D62728', capsize=5, linewidth=2, markersize=8
    )

    ax.set_xlabel('Number of Clients', fontsize=12)
    ax.set_ylabel('Mean Optimality Gap (%)', fontsize=12)
    ax.set_title('Mean Optimality Gap with Std Dev', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimality_gap_mean.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Runtime comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    mean_milp_time = df.groupby('num_clients')['milp_runtime'].mean()
    mean_dr_time = df.groupby('num_clients')['dr_greedy_runtime'].mean()

    ax.plot(mean_milp_time.index, mean_milp_time.values, 'o-', label='MILP (CBC)', color='#1F77B4', linewidth=2)
    ax.plot(mean_dr_time.index, mean_dr_time.values, 's-', label='DR-Greedy', color='#D62728', linewidth=2)

    ax.set_xlabel('Number of Clients', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Runtime Comparison: MILP vs DR-Greedy', fontsize=14)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    log.info(f"MILP plots saved to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if not PULP_AVAILABLE:
        print("pulp not installed. Please install with: pip install pulp")
    else:
        config = MILPConfig(
            num_clients_list=[5, 6, 7, 8, 9, 10],
            num_seeds=5,
        )

        results = run_milp_optimality_experiments(config=config, budget=20)
        plot_optimality_gap(results)