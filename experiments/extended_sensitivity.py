"""
Extended Sensitivity Experiments (Requirement 4).

Sweeps additional parameters:
- num_clients: [10, 20, 30, 40, 50]
- num_satellites: [1, 2, 3, 4]
- CVaR target: [0.90, 0.95, 0.99]
- timing/phase error std: [0.01, 0.05, 0.1, 0.2] radians
"""

from __future__ import annotations
import csv
import logging
import os
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation.topology.nodes import Node, NodeType, generate_nodes
from simulation.config_loader import load_config
from optimization.baselines import (
    lop_selection, go_selection, nrs_selection, random_selection,
    da_selection, fedsn_selection, hsfl_selection, dr_selection
)
from optimization.placement import greedy_server_selection, run_inner_ota_loop
from optimization.objective import compute_objective
from simulation.run_simulation import build_costs
from simulation.topology.aircomp import compute_amse_hierarchical
from simulation.evaluation.metrics import compute_e2e_latency, compute_total_energy
from skyfield.api import load

log = logging.getLogger(__name__)


@dataclass
class SensitivityConfig:
    """Configuration for extended sensitivity experiments."""
    num_clients_list: List[int]
    num_satellites_list: List[int]
    cvar_alpha_list: List[float]
    timing_error_std_list: List[float]
    num_seeds: int
    output_dir: str = "results/experiments/sensitivity"


@dataclass
class SensitivityResult:
    """Single sensitivity experiment result."""
    experiment_type: str  # 'num_clients', 'num_satellites', 'cvar_alpha', 'timing_error'
    param_value: float
    seed: int
    algorithm: str
    latency: float
    amse: float
    energy: float
    cvar_90: float
    cvar_95: float
    cvar_99: float
    runtime: float
    num_sat_selected: int
    num_uav_selected: int
    num_ground_selected: int


ALL_ALGORITHMS = {
    "lop": lop_selection,
    "go": go_selection,
    "nrs": nrs_selection,
    "random": random_selection,
    "da": da_selection,
    "fedsn": fedsn_selection,
    "hsfl": hsfl_selection,
    "dr_greedy": dr_selection,
}


def run_sensitivity_single(
    experiment_type: str,
    param_value: float,
    algorithm_name: str,
    algorithm_func,
    candidates: List[Node],
    clients: List[Node],
    budget: int,
    cost: Dict[Node, float],
    thresh: float,
    alpha: float,
    beta: float,
    delta_list: List[float],
    num_scenarios: int,
    t_now,
    cvar_alpha: float = 0.95,
    timing_error_std: float = 0.0,
    seed: int = 42,
) -> SensitivityResult:
    """Run a single sensitivity experiment configuration."""
    start_time = time.perf_counter()
    np.random.seed(seed)

    # Apply timing/phase error to SNR if specified
    if timing_error_std > 0:
        # Simulate timing/phase errors by perturbing SNR
        for cl in clients:
            for s in candidates:
                # We'll handle this in the algorithm via noisy SNR
                pass

    # Run algorithm
    if algorithm_name == "dr_greedy":
        selected = algorithm_func(
            candidates=candidates, clients=clients, budget=budget, cost=cost,
            thresh=thresh, alpha=alpha, beta=beta, delta_list=delta_list,
            N=num_scenarios, t_now=t_now, epsilon=0.1, kappa=0.3,
            cvar_alpha=cvar_alpha
        )
    else:
        selected = algorithm_func(
            candidates=candidates, clients=clients, budget=budget, cost=cost,
            thresh=thresh, alpha=alpha, beta=beta, delta_list=delta_list,
            N=num_scenarios, t_now=t_now
        )

    # Compute metrics
    if selected and clients:
        latency = compute_e2e_latency(clients, selected, t_now)

        tier_sync_errors = {1: 1e-9, 2: 5e-9, 3: 1e-8}
        # Add timing error to sync errors if specified
        if timing_error_std > 0:
            for k in tier_sync_errors:
                tier_sync_errors[k] += timing_error_std

        amse = compute_amse_hierarchical(
            selected, clients, delta_list, t_now=t_now,
            tier_sync_errors=tier_sync_errors
        )

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

        # CVaR computation
        from optimization.dro import cvar
        # Generate sample losses
        sample_losses = []
        for _ in range(100):
            losses = []
            for cl in clients:
                for s in selected:
                    snr = cl.compute_snr_to(s, t_now)
                    if snr > thresh:
                        loss_val = 1.0 / (1.0 + snr)
                        losses.append(loss_val)
            if losses:
                sample_losses.append(np.mean(losses))
        if sample_losses:
            cvar_90 = cvar(np.array(sample_losses), alpha=0.90)
            cvar_95 = cvar(np.array(sample_losses), alpha=0.95)
            cvar_99 = cvar(np.array(sample_losses), alpha=0.99)
        else:
            cvar_90 = cvar_95 = cvar_99 = 0.0
    else:
        latency = float('inf')
        amse = float('inf')
        energy = float('inf')
        cvar_90 = cvar_95 = cvar_99 = float('inf')

    runtime = time.perf_counter() - start_time

    num_sat = sum(1 for s in selected if s.type == NodeType.SATELLITE)
    num_uav = sum(1 for s in selected if s.type == NodeType.UAV)
    num_ground = sum(1 for s in selected if s.type == NodeType.GROUND)

    return SensitivityResult(
        experiment_type=experiment_type,
        param_value=param_value,
        seed=seed,
        algorithm=algorithm_name,
        latency=latency,
        amse=amse,
        energy=energy,
        cvar_90=cvar_90,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        runtime=runtime,
        num_sat_selected=num_sat,
        num_uav_selected=num_uav,
        num_ground_selected=num_ground,
    )


def run_extended_sensitivity(
    config: SensitivityConfig = None,
    budget: int = 20,
    thresh: float = 0.0,
    delta_list: List[float] = None,
    num_scenarios: int = 16,
    alpha: float = 0.5,
    beta: float = 0.5,
    seeds: List[int] = None,
) -> List[SensitivityResult]:
    """
    Run extended sensitivity experiments across multiple dimensions.
    """
    if config is None:
        config = SensitivityConfig(
            num_clients_list=[10, 20, 30, 40, 50],
            num_satellites_list=[1, 2, 3, 4],
            cvar_alpha_list=[0.90, 0.95, 0.99],
            timing_error_std_list=[0.01, 0.05, 0.1, 0.2],
            num_seeds=3,
        )

    if delta_list is None:
        delta_list = [0.1, 0.2]

    if seeds is None:
        seeds = list(range(config.num_seeds))

    base_config = load_config("configs/default.yaml")
    sim_config = base_config["simulation"]
    base_num_sats = sim_config.get("num_sats", 1)
    base_num_uavs = sim_config.get("num_uavs", 2)
    base_num_ground = sim_config.get("num_ground", 4)
    area_size = sim_config.get("area_size", 2000)
    gradient_dim = sim_config.get("gradient_dim", 100)

    os.makedirs(config.output_dir, exist_ok=True)
    csv_path = os.path.join(config.output_dir, "extended_sensitivity.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'experiment_type', 'param_value', 'seed', 'algorithm',
            'latency', 'amse', 'energy', 'cvar_90', 'cvar_95', 'cvar_99',
            'runtime', 'num_sat_selected', 'num_uav_selected', 'num_ground_selected'
        ])

    all_results = []

    # Experiment 1: Vary num_clients
    log.info("=== Experiment 1: Varying num_clients ===")
    for num_clients in config.num_clients_list:
        for seed in seeds:
            ts = load.timescale()
            t_now = ts.now()

            nodes = generate_nodes(
                num_sats=base_num_sats,
                num_uavs=base_num_uavs,
                num_ground=base_num_ground,
                num_clients=num_clients,
                area_size=area_size,
                gradient_dim=gradient_dim,
                t0=t_now
            )

            clients = [n for n in nodes if n.type == NodeType.CLIENT]
            candidates = [n for n in nodes if n.type != NodeType.CLIENT]
            cost = build_costs(candidates)

            for algo_name, algo_func in ALL_ALGORITHMS.items():
                result = run_sensitivity_single(
                    experiment_type='num_clients',
                    param_value=num_clients,
                    algorithm_name=algo_name,
                    algorithm_func=algo_func,
                    candidates=candidates,
                    clients=clients,
                    budget=budget,
                    cost=cost,
                    thresh=thresh,
                    alpha=alpha,
                    beta=beta,
                    delta_list=delta_list,
                    num_scenarios=num_scenarios,
                    t_now=t_now,
                    seed=seed,
                )
                all_results.append(result)
                _append_to_csv(csv_path, result)

    # Experiment 2: Vary num_satellites
    log.info("=== Experiment 2: Varying num_satellites ===")
    for num_sats in config.num_satellites_list:
        for seed in seeds:
            ts = load.timescale()
            t_now = ts.now()

            nodes = generate_nodes(
                num_sats=num_sats,
                num_uavs=base_num_uavs,
                num_ground=base_num_ground,
                num_clients=20,  # Fixed client count
                area_size=area_size,
                gradient_dim=gradient_dim,
                t0=t_now
            )

            clients = [n for n in nodes if n.type == NodeType.CLIENT]
            candidates = [n for n in nodes if n.type != NodeType.CLIENT]
            cost = build_costs(candidates)

            for algo_name, algo_func in ALL_ALGORITHMS.items():
                result = run_sensitivity_single(
                    experiment_type='num_satellites',
                    param_value=num_sats,
                    algorithm_name=algo_name,
                    algorithm_func=algo_func,
                    candidates=candidates,
                    clients=clients,
                    budget=budget,
                    cost=cost,
                    thresh=thresh,
                    alpha=alpha,
                    beta=beta,
                    delta_list=delta_list,
                    num_scenarios=num_scenarios,
                    t_now=t_now,
                    seed=seed,
                )
                all_results.append(result)
                _append_to_csv(csv_path, result)

    # Experiment 3: Vary CVaR alpha
    log.info("=== Experiment 3: Varying CVaR alpha ===")
    for cvar_alpha in config.cvar_alpha_list:
        for seed in seeds:
            ts = load.timescale()
            t_now = ts.now()

            nodes = generate_nodes(
                num_sats=base_num_sats,
                num_uavs=base_num_uavs,
                num_ground=base_num_ground,
                num_clients=20,
                area_size=area_size,
                gradient_dim=gradient_dim,
                t0=t_now
            )

            clients = [n for n in nodes if n.type == NodeType.CLIENT]
            candidates = [n for n in nodes if n.type != NodeType.CLIENT]
            cost = build_costs(candidates)

            for algo_name, algo_func in ALL_ALGORITHMS.items():
                result = run_sensitivity_single(
                    experiment_type='cvar_alpha',
                    param_value=cvar_alpha,
                    algorithm_name=algo_name,
                    algorithm_func=algo_func,
                    candidates=candidates,
                    clients=clients,
                    budget=budget,
                    cost=cost,
                    thresh=thresh,
                    alpha=alpha,
                    beta=beta,
                    delta_list=delta_list,
                    num_scenarios=num_scenarios,
                    t_now=t_now,
                    cvar_alpha=cvar_alpha,
                    seed=seed,
                )
                all_results.append(result)
                _append_to_csv(csv_path, result)

    # Experiment 4: Vary timing/phase error
    log.info("=== Experiment 4: Varying timing/phase error ===")
    for timing_error in config.timing_error_std_list:
        for seed in seeds:
            ts = load.timescale()
            t_now = ts.now()

            nodes = generate_nodes(
                num_sats=base_num_sats,
                num_uavs=base_num_uavs,
                num_ground=base_num_ground,
                num_clients=20,
                area_size=area_size,
                gradient_dim=gradient_dim,
                t0=t_now
            )

            clients = [n for n in nodes if n.type == NodeType.CLIENT]
            candidates = [n for n in nodes if n.type != NodeType.CLIENT]
            cost = build_costs(candidates)

            for algo_name, algo_func in ALL_ALGORITHMS.items():
                result = run_sensitivity_single(
                    experiment_type='timing_error',
                    param_value=timing_error,
                    algorithm_name=algo_name,
                    algorithm_func=algo_func,
                    candidates=candidates,
                    clients=clients,
                    budget=budget,
                    cost=cost,
                    thresh=thresh,
                    alpha=alpha,
                    beta=beta,
                    delta_list=delta_list,
                    num_scenarios=num_scenarios,
                    t_now=t_now,
                    timing_error_std=timing_error,
                    seed=seed,
                )
                all_results.append(result)
                _append_to_csv(csv_path, result)

    log.info(f"Extended sensitivity experiments complete. Results saved to {csv_path}")
    return all_results


def _append_to_csv(csv_path: str, result: SensitivityResult):
    """Append a single result to CSV."""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            result.experiment_type, result.param_value, result.seed, result.algorithm,
            result.latency, result.amse, result.energy,
            result.cvar_90, result.cvar_95, result.cvar_99,
            result.runtime,
            result.num_sat_selected, result.num_uav_selected, result.num_ground_selected
        ])


def plot_sensitivity_results(results: List[SensitivityResult], output_dir: str = "plots/experiments/sensitivity"):
    """Generate sensitivity analysis plots."""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame([r.__dict__ for r in results])

    if df.empty:
        log.warning("No results for sensitivity plotting")
        return

    # Filter finite values
    metrics = ['latency', 'amse', 'energy', 'cvar_90', 'cvar_95', 'cvar_99']
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce')

    experiment_types = df['experiment_type'].unique()

    colors = {
        "dr_greedy": "#D62728", "da": "#1F77B4", "lop": "#2CA02C",
        "go": "#9467BD", "nrs": "#8C564B", "random": "#17BECF",
        "fedsn": "#FF7F0E", "hsfl": "#7F7F7F",
    }
    markers = {
        "dr_greedy": "o", "da": "s", "lop": "^", "go": "D",
        "nrs": "v", "random": "P", "fedsn": "X", "hsfl": "*",
    }

    for exp_type in experiment_types:
        exp_df = df[df['experiment_type'] == exp_type].copy()

        if exp_df.empty:
            continue

        # Group by param_value and algorithm, average over seeds
        agg = exp_df.groupby(['param_value', 'algorithm']).agg({
            'latency': 'mean', 'amse': 'mean', 'energy': 'mean',
            'cvar_90': 'mean', 'cvar_95': 'mean', 'cvar_99': 'mean'
        }).reset_index()

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))

            for algo in ALL_ALGORITHMS.keys():
                algo_df = agg[agg['algorithm'] == algo]
                if algo_df.empty:
                    continue

                x = algo_df['param_value'].values
                y = algo_df[metric].values

                # Sort by x
                sort_idx = np.argsort(x)
                x = x[sort_idx]
                y = y[sort_idx]

                ax.plot(x, y, marker=markers.get(algo, 'o'),
                       color=colors.get(algo, 'black'), label=algo,
                       linewidth=2, markersize=8, alpha=0.8)

            ax.set_xlabel(exp_type.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f'Sensitivity: {metric.upper()} vs {exp_type.replace("_", " ").title()}', fontsize=14)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

            if metric in ['amse', 'cvar_90', 'cvar_95', 'cvar_99']:
                ax.set_yscale('log')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sensitivity_{exp_type}_{metric}.png"), dpi=300, bbox_inches='tight')
            plt.close()

    log.info(f"Sensitivity plots saved to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = SensitivityConfig(
        num_clients_list=[10, 20, 30, 40, 50],
        num_satellites_list=[1, 2, 3, 4],
        cvar_alpha_list=[0.90, 0.95, 0.99],
        timing_error_std_list=[0.01, 0.05, 0.1, 0.2],
        num_seeds=3,
    )

    results = run_extended_sensitivity(config=config, budget=20)
    plot_sensitivity_results(results)