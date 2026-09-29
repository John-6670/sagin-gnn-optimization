"""
Pareto Front Experiments (Requirement 2).

Sweeps (ωL, ωA, Amax, ε) for all methods and plots latency–AMSE–energy fronts
instead of single operating points.
"""

from __future__ import annotations
import csv
import logging
import os
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from simulation.topology.nodes import Node, NodeType, generate_nodes
from simulation.config_loader import load_config
from optimization.baselines import (
    lop_selection, go_selection, nrs_selection, random_selection,
    da_selection, fedsn_selection, hsfl_selection, dr_selection
)
from optimization.placement import greedy_server_selection
from optimization.objective import compute_objective
from simulation.run_simulation import build_costs
from simulation.topology.aircomp import compute_amse_hierarchical
from simulation.evaluation.metrics import compute_e2e_latency, compute_total_energy
from skyfield.api import load

log = logging.getLogger(__name__)


@dataclass
class ParetoConfig:
    """Configuration for Pareto front experiments."""
    omega_l_values: List[float]
    amax_values: List[int]
    epsilon_values: List[float]
    num_seeds: int
    output_dir: str = "results/experiments/pareto"


@dataclass
class ParetoResult:
    """Single Pareto experiment result."""
    algorithm: str
    seed: int
    omega_l: float
    amax: int
    epsilon: float
    latency: float
    amse: float
    energy: float
    runtime: float
    num_sat: int
    num_uav: int
    num_ground: int


# All algorithms to test
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


def get_pareto_mask_3d(points: np.ndarray) -> np.ndarray:
    """
    Return boolean mask identifying Pareto-optimal points in 3D.
    All objectives are minimized.
    """
    is_pareto = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if not is_pareto[i]:
            continue
        dominated = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
        if np.any(dominated):
            is_pareto[i] = False
    return is_pareto




def run_pareto_seed_worker(
    seed: int,
    config: ParetoConfig,
    budget: int,
    thresh: float,
    delta_list: List[float],
    num_scenarios: int,
    output_dir: str,
) -> List[ParetoResult]:
    """Worker function: run all Pareto configs for ONE seed (full topology + all algos)."""
    import numpy as np
    import time
    import csv
    import os
    from simulation.topology.nodes import NodeType, generate_nodes
    from simulation.config_loader import load_config
    from optimization.baselines import (
        lop_selection, go_selection, nrs_selection, random_selection,
        da_selection, fedsn_selection, hsfl_selection, dr_selection
    )
    from simulation.evaluation.metrics import compute_e2e_latency, compute_total_energy
    from simulation.topology.aircomp import compute_amse_hierarchical
    from optimization.placement import run_inner_ota_loop
    from simulation.run_simulation import build_costs
    from skyfield.api import load

    # Load base config for topology params
    base_config = load_config("configs/default.yaml")
    sim_config = base_config["simulation"]
    num_sats = sim_config.get("num_sats", 1)
    num_uavs = sim_config.get("num_uavs", 2)
    num_ground = sim_config.get("num_ground", 4)
    num_clients = sim_config.get("num_clients", 20)
    area_size = sim_config.get("area_size", 2000)
    gradient_dim = sim_config.get("gradient_dim", 100)

    # Generate topology for this seed
    ts = load.timescale()
    t_now = ts.now()

    np.random.seed(seed)
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

    all_results = []
    csv_path = os.path.join(output_dir, f"pareto_results_seed_{seed}.csv")

    # Write header for this seed's CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'algorithm', 'seed', 'omega_l', 'amax', 'epsilon',
            'latency', 'amse', 'energy', 'runtime',
            'num_sat', 'num_uav', 'num_ground'
        ])

    for omega_l in config.omega_l_values:
        for amax in config.amax_values:
            for epsilon in config.epsilon_values:
                for algo_name, algo_func in ALL_ALGORITHMS.items():
                    start_time = time.perf_counter()
                    np.random.seed(seed)

                    eff_alpha = omega_l
                    eff_beta = 1.0 - omega_l

                    try:
                        if algo_name == "dr_greedy":
                            selected = algo_func(
                                candidates=candidates, clients=clients, budget=budget, cost=cost,
                                thresh=thresh, alpha=eff_alpha, beta=eff_beta, delta_list=delta_list,
                                N=num_scenarios, t_now=t_now, epsilon=epsilon, kappa=0.3
                            )
                        else:
                            selected = algo_func(
                                candidates=candidates, clients=clients, budget=budget, cost=cost,
                                thresh=thresh, alpha=eff_alpha, beta=eff_beta, delta_list=delta_list,
                                N=num_scenarios, t_now=t_now
                            )

                        # Compute metrics
                        if selected and clients:
                            latency = compute_e2e_latency(clients, selected, t_now)

                            tier_sync_errors = {1: 1e-9, 2: 5e-9, 3: 1e-8}
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
                        else:
                            latency = float('inf')
                            amse = float('inf')
                            energy = float('inf')

                        runtime = time.perf_counter() - start_time

                        num_sat = sum(1 for s in selected if s.type == NodeType.SATELLITE)
                        num_uav = sum(1 for s in selected if s.type == NodeType.UAV)
                        num_ground = sum(1 for s in selected if s.type == NodeType.GROUND)

                        result = ParetoResult(
                            algorithm=algo_name,
                            seed=seed,
                            omega_l=omega_l,
                            amax=amax,
                            epsilon=epsilon,
                            latency=latency,
                            amse=amse,
                            energy=energy,
                            runtime=runtime,
                            num_sat=num_sat,
                            num_uav=num_uav,
                            num_ground=num_ground,
                        )
                        all_results.append(result)

                        # Append to seed CSV
                        with open(csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                result.algorithm, result.seed, result.omega_l,
                                result.amax, result.epsilon,
                                result.latency, result.amse, result.energy,
                                result.runtime,
                                result.num_sat, result.num_uav, result.num_ground
                            ])
                    except Exception as e:
                        import logging
                        log = logging.getLogger(__name__)
                        log.error(f"Error in {algo_name} (seed={seed}, ωL={omega_l}, Amax={amax}, ε={epsilon}): {e}")
                        continue

    return all_results


def run_pareto_experiments(
    config: ParetoConfig = None,
    budget: int = 20,
    thresh: float = 0.0,
    delta_list: List[float] = None,
    num_scenarios: int = 16,
    seeds: List[int] = None,
) -> List[ParetoResult]:
    """
    Run full Pareto front experiments sweeping parameters.
    Parallelized across seeds (each seed generates its own topology and runs all algos).

    Args:
        config: ParetoConfig with sweep parameters
        budget: Budget constraint
        thresh: SNR threshold
        delta_list: Quantization deltas
        num_scenarios: Number of DRO scenarios
        seeds: List of random seeds

    Returns:
        List of ParetoResult objects
    """
    if config is None:
        config = ParetoConfig(
            omega_l_values=[0.1, 0.3, 0.5, 0.7, 0.9],
            amax_values=[10, 20, 30, 40],
            epsilon_values=[0.05, 0.1, 0.2, 0.5],
            num_seeds=3,
        )

    if delta_list is None:
        delta_list = [0.1, 0.2]

    if seeds is None:
        seeds = list(range(config.num_seeds))

    os.makedirs(config.output_dir, exist_ok=True)
    csv_path = os.path.join(config.output_dir, "pareto_results.csv")

    # Write header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'algorithm', 'seed', 'omega_l', 'amax', 'epsilon',
            'latency', 'amse', 'energy', 'runtime',
            'num_sat', 'num_uav', 'num_ground'
        ])

    all_results = []
    total_configs = len(config.omega_l_values) * len(config.amax_values) * \
                    len(config.epsilon_values) * len(ALL_ALGORITHMS) * len(seeds)

    log.info(f"Starting Pareto experiments: {total_configs} total configurations")

    # Parallel across seeds (each worker generates topology + runs all algos)
    max_workers = min(len(seeds), mp.cpu_count())
    log.info(f"Using {max_workers} parallel workers across {len(seeds)} seeds")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_seed = {
            executor.submit(
                run_pareto_seed_worker,
                seed, config, budget, thresh, delta_list, num_scenarios, config.output_dir
            ): seed
            for seed in seeds
        }

        completed_configs = 0
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                seed_results = future.result()
                all_results.extend(seed_results)
                completed_configs += len(seed_results)
                log.info(f"Seed {seed} complete: {len(seed_results)} runs. Total: {completed_configs}/{total_configs}")

                # Merge seed CSV into main CSV
                seed_csv = os.path.join(config.output_dir, f"pareto_results_seed_{seed}.csv")
                if os.path.exists(seed_csv):
                    with open(seed_csv, 'r') as sf, open(csv_path, 'a', newline='') as mf:
                        reader = csv.reader(sf)
                        writer = csv.writer(mf)
                        next(reader)  # skip header
                        for row in reader:
                            writer.writerow(row)
            except Exception as e:
                log.error(f"Error in seed {seed}: {e}")
                continue

    log.info(f"Pareto experiments complete. Results saved to {csv_path}")
    return all_results


def plot_pareto_fronts_3d(results: List[ParetoResult], output_dir: str = "plots/experiments/pareto"):
    """Generate 3D Pareto front plots."""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame([r.__dict__ for r in results])

    # Filter finite values
    df = df[np.isfinite(df['latency']) & np.isfinite(df['amse']) & np.isfinite(df['energy'])]

    if df.empty:
        log.warning("No finite results for Pareto plotting")
        return

    # Get unique algorithms
    algorithms = df['algorithm'].unique()

    # Color mapping
    colors = {
        "dr_greedy": "#D62728", "da": "#1F77B4", "lop": "#2CA02C",
        "go": "#9467BD", "nrs": "#8C564B", "random": "#17BECF",
        "fedsn": "#FF7F0E", "hsfl": "#7F7F7F",
    }
    markers = {
        "dr_greedy": "o", "da": "s", "lop": "^", "go": "D",
        "nrs": "v", "random": "P", "fedsn": "X", "hsfl": "*",
    }

    # 3D Scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for algo in algorithms:
        algo_df = df[df['algorithm'] == algo]
        if algo_df.empty:
            continue

        # Aggregate by taking mean across seeds for each param combination
        agg = algo_df.groupby(['omega_l', 'amax', 'epsilon']).agg({
            'latency': 'mean', 'amse': 'mean', 'energy': 'mean'
        }).reset_index()

        ax.scatter(
            agg['latency'], agg['amse'], agg['energy'],
            c=colors.get(algo, 'black'),
            marker=markers.get(algo, 'o'),
            s=50, alpha=0.6, label=algo, edgecolors='black', linewidths=0.3
        )

        # Highlight Pareto front
        points = agg[['latency', 'amse', 'energy']].values
        pareto_mask = get_pareto_mask_3d(points)
        pareto_points = agg[pareto_mask]

        ax.scatter(
            pareto_points['latency'], pareto_points['amse'], pareto_points['energy'],
            facecolors='none', edgecolors=colors.get(algo, 'black'),
            marker='o', s=150, linewidths=2, zorder=10
        )

    ax.set_xlabel('Latency', fontsize=12)
    ax.set_ylabel('AMSE', fontsize=12)
    ax.set_zlabel('Energy', fontsize=12)
    ax.set_title('3D Pareto Front: Latency vs AMSE vs Energy', fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_3d_latency_amse_energy.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2D Projections
    projections = [
        ('latency', 'amse', 'Latency', 'AMSE', 'pareto_latency_amse.png'),
        ('latency', 'energy', 'Latency', 'Energy', 'pareto_latency_energy.png'),
        ('amse', 'energy', 'AMSE', 'Energy', 'pareto_amse_energy.png'),
    ]

    for x_col, y_col, x_label, y_label, fname in projections:
        fig, ax = plt.subplots(figsize=(8, 6))

        for algo in algorithms:
            algo_df = df[df['algorithm'] == algo]
            if algo_df.empty:
                continue

            agg = algo_df.groupby(['omega_l', 'amax', 'epsilon']).agg({
                x_col: 'mean', y_col: 'mean'
            }).reset_index()

            ax.scatter(
                agg[x_col], agg[y_col],
                c=colors.get(algo, 'black'),
                marker=markers.get(algo, 'o'),
                s=50, alpha=0.6, label=algo, edgecolors='black', linewidths=0.3
            )

            # Pareto front
            points = agg[[x_col, y_col]].values
            pareto_mask = get_pareto_mask_3d(points)
            pareto_points = agg[pareto_mask].sort_values(x_col)

            ax.plot(
                pareto_points[x_col], pareto_points[y_col],
                color=colors.get(algo, 'black'), linestyle='--', linewidth=2, alpha=0.8
            )

            ax.scatter(
                pareto_points[x_col], pareto_points[y_col],
                facecolors='none', edgecolors=colors.get(algo, 'black'),
                marker='o', s=120, linewidths=2, zorder=10
            )

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'Pareto Front: {x_label} vs {y_label}', fontsize=14)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        if 'amse' in y_col or 'amse' in x_col:
            ax.set_yscale('log')
        if 'energy' in y_col or 'energy' in x_col:
            ax.set_xscale('log') if x_col == 'energy' else ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()

    log.info(f"Pareto plots saved to {output_dir}")


def load_pareto_results(csv_path: str) -> List[ParetoResult]:
    """Load Pareto results from CSV."""
    df = pd.read_csv(csv_path)
    return [ParetoResult(**row) for _, row in df.iterrows()]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = ParetoConfig(
        omega_l_values=[0.1, 0.3, 0.5, 0.7, 0.9],
        amax_values=[10, 20, 30, 40],
        epsilon_values=[0.05, 0.1, 0.2, 0.5],
        num_seeds=3,
    )

    results = run_pareto_experiments(config=config, budget=20)
    plot_pareto_fronts_3d(results)