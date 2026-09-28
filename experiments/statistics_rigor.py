"""
Statistics Rigor Experiments (Requirement 5).

Runs ≥5 seeds per configuration, reports 95% CIs via bootstrap,
and provides detailed runtime measurements.
"""

from __future__ import annotations
import csv
import logging
import os
import time
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from simulation.topology.nodes import Node, NodeType, generate_nodes
from simulation.config_loader import load_config
from optimization.baselines import (
    lop_selection, go_selection, nrs_selection, random_selection,
    da_selection, fedsn_selection, hsfl_selection, dr_selection
)
from optimization.placement import greedy_server_selection, run_inner_ota_loop
from simulation.run_simulation import build_costs
from simulation.topology.aircomp import compute_amse_hierarchical
from simulation.evaluation.metrics import compute_e2e_latency, compute_total_energy
from skyfield.api import load

log = logging.getLogger(__name__)


@dataclass
class StatisticsConfig:
    """Configuration for statistical rigor experiments."""
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 999, 111, 222, 333, 444, 555])
    num_bootstrap: int = 10000
    confidence_level: float = 0.95
    output_dir: str = "results/experiments/statistics"
    measure_runtime: bool = True


@dataclass
class StatResult:
    """Statistical result for a single metric."""
    algorithm: str
    metric: str
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    median: float
    q25: float
    q75: float
    min_val: float
    max_val: float
    n_samples: int


@dataclass
class RuntimeResult:
    """Runtime measurement result."""
    algorithm: str
    mean_time: float
    std_time: float
    ci_lower: float
    ci_upper: float
    median_time: float
    min_time: float
    max_time: float
    n_samples: int


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


def bootstrap_ci(data: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 10000) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if len(data) < 2:
        return data[0], data[0] if len(data) == 1 else (np.nan, np.nan)

    bootstrap_means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return float(lower), float(upper)


def compute_statistics(values: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 10000) -> Dict:
    """Compute comprehensive statistics for an array of values."""
    finite_vals = values[np.isfinite(values)]

    if len(finite_vals) == 0:
        return {
            'mean': np.nan, 'std': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
            'median': np.nan, 'q25': np.nan, 'q75': np.nan,
            'min': np.nan, 'max': np.nan, 'n': 0
        }

    ci_lower, ci_upper = bootstrap_ci(finite_vals, confidence, n_bootstrap)

    return {
        'mean': float(np.mean(finite_vals)),
        'std': float(np.std(finite_vals, ddof=1)),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'median': float(np.median(finite_vals)),
        'q25': float(np.percentile(finite_vals, 25)),
        'q75': float(np.percentile(finite_vals, 75)),
        'min': float(np.min(finite_vals)),
        'max': float(np.max(finite_vals)),
        'n': len(finite_vals),
    }


def run_single_stat_experiment(
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
    seed: int,
) -> Dict[str, float]:
    """Run a single experiment and return all metrics + runtime."""
    np.random.seed(seed)

    start_time = time.perf_counter()

    if algorithm_name == "dr_greedy":
        selected = algorithm_func(
            candidates=candidates, clients=clients, budget=budget, cost=cost,
            thresh=thresh, alpha=alpha, beta=beta, delta_list=delta_list,
            N=num_scenarios, t_now=t_now, epsilon=0.1, kappa=0.3
        )
    else:
        selected = algorithm_func(
            candidates=candidates, clients=clients, budget=budget, cost=cost,
            thresh=thresh, alpha=alpha, beta=beta, delta_list=delta_list,
            N=num_scenarios, t_now=t_now
        )

    runtime = time.perf_counter() - start_time

    metrics = {'runtime': runtime}

    if selected and clients:
        metrics['latency'] = compute_e2e_latency(clients, selected, t_now)

        tier_sync_errors = {1: 1e-9, 2: 5e-9, 3: 1e-8}
        metrics['amse'] = compute_amse_hierarchical(
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
        metrics['energy'] = compute_total_energy(clients, selected, ota, transmission_time=10, t_now=t_now)

        # Server composition
        metrics['num_sat'] = sum(1 for s in selected if s.type == NodeType.SATELLITE)
        metrics['num_uav'] = sum(1 for s in selected if s.type == NodeType.UAV)
        metrics['num_ground'] = sum(1 for s in selected if s.type == NodeType.GROUND)
    else:
        metrics.update({
            'latency': np.inf, 'amse': np.inf, 'energy': np.inf,
            'num_sat': 0, 'num_uav': 0, 'num_ground': 0
        })

    return metrics


def run_statistical_experiments(
    config: StatisticsConfig = None,
    budget: int = 20,
    thresh: float = 0.0,
    delta_list: List[float] = None,
    num_scenarios: int = 16,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Tuple[List[StatResult], List[RuntimeResult]]:
    """
    Run statistical rigor experiments with multiple seeds.
    """
    if config is None:
        config = StatisticsConfig()

    if delta_list is None:
        delta_list = [0.1, 0.2]

    base_config = load_config("configs/default.yaml")
    sim_config = base_config["simulation"]
    num_sats = sim_config.get("num_sats", 1)
    num_uavs = sim_config.get("num_uavs", 2)
    num_ground = sim_config.get("num_ground", 4)
    num_clients = sim_config.get("num_clients", 20)
    area_size = sim_config.get("area_size", 2000)
    gradient_dim = sim_config.get("gradient_dim", 100)

    os.makedirs(config.output_dir, exist_ok=True)

    # Per-seed results CSV
    seed_csv_path = os.path.join(config.output_dir, "per_seed_results.csv")
    with open(seed_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'algorithm', 'seed', 'latency', 'amse', 'energy',
            'num_sat', 'num_uav', 'num_ground', 'runtime'
        ])

    # Summary statistics CSV
    stats_csv_path = os.path.join(config.output_dir, "statistics_summary.csv")
    with open(stats_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'algorithm', 'metric', 'mean', 'std', 'ci_lower', 'ci_upper',
            'median', 'q25', 'q75', 'min', 'max', 'n'
        ])

    # Runtime CSV
    runtime_csv_path = os.path.join(config.output_dir, "runtime_statistics.csv")
    with open(runtime_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'algorithm', 'mean_time', 'std_time', 'ci_lower', 'ci_upper',
            'median_time', 'min_time', 'max_time', 'n'
        ])

    all_seed_results = {algo: [] for algo in ALL_ALGORITHMS}
    all_runtimes = {algo: [] for algo in ALL_ALGORITHMS}

    total_configs = len(ALL_ALGORITHMS) * len(config.seeds)
    config_idx = 0

    for seed in config.seeds:
        ts = load.timescale()
        t_now = ts.now()

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

        for algo_name, algo_func in ALL_ALGORITHMS.items():
            config_idx += 1
            log.info(f"Progress: {config_idx}/{total_configs} - {algo_name}, seed={seed}")

            try:
                metrics = run_single_stat_experiment(
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

                # Store per-seed results
                all_seed_results[algo_name].append(metrics)
                all_runtimes[algo_name].append(metrics['runtime'])

                # Write per-seed row
                with open(seed_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        algo_name, seed,
                        metrics['latency'], metrics['amse'], metrics['energy'],
                        metrics['num_sat'], metrics['num_uav'], metrics['num_ground'],
                        metrics['runtime']
                    ])

            except Exception as e:
                log.error(f"Error in {algo_name} (seed={seed}): {e}")
                continue

    # Compute and write statistics
    stat_results = []
    runtime_results = []

    for algo_name in ALL_ALGORITHMS:
        results = all_seed_results[algo_name]
        runtimes = all_runtimes[algo_name]

        if not results:
            continue

        # Statistics for each metric
        metrics_to_analyze = ['latency', 'amse', 'energy', 'num_sat', 'num_uav', 'num_ground']
        for metric in metrics_to_analyze:
            values = np.array([r[metric] for r in results])
            stats_dict = compute_statistics(values, config.confidence_level, config.num_bootstrap)

            stat_result = StatResult(
                algorithm=algo_name,
                metric=metric,
                **stats_dict
            )
            stat_results.append(stat_result)

            with open(stats_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    stat_result.algorithm, stat_result.metric,
                    stat_result.mean, stat_result.std,
                    stat_result.ci_lower, stat_result.ci_upper,
                    stat_result.median, stat_result.q25, stat_result.q75,
                    stat_result.min_val, stat_result.max_val, stat_result.n_samples
                ])

        # Runtime statistics
        runtime_values = np.array(runtimes)
        runtime_stats = compute_statistics(runtime_values, config.confidence_level, config.num_bootstrap)

        runtime_result = RuntimeResult(
            algorithm=algo_name,
            mean_time=runtime_stats['mean'],
            std_time=runtime_stats['std'],
            ci_lower=runtime_stats['ci_lower'],
            ci_upper=runtime_stats['ci_upper'],
            median_time=runtime_stats['median'],
            min_time=runtime_stats['min'],
            max_time=runtime_stats['max'],
            n_samples=runtime_stats['n'],
        )
        runtime_results.append(runtime_result)

        with open(runtime_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                runtime_result.algorithm, runtime_result.mean_time,
                runtime_result.std_time, runtime_result.ci_lower,
                runtime_result.ci_upper, runtime_result.median_time,
                runtime_result.min_time, runtime_result.max_time,
                runtime_result.n_samples
            ])

    log.info(f"Statistical experiments complete. Results saved to {config.output_dir}")
    return stat_results, runtime_results


def plot_statistics(stat_results: List[StatResult], runtime_results: List[RuntimeResult],
                   config: StatisticsConfig,
                   output_dir: str = "plots/experiments/statistics"):
    """Generate statistical rigor plots with 95% CIs."""
    os.makedirs(output_dir, exist_ok=True)

    stat_df = pd.DataFrame([r.__dict__ for r in stat_results])
    runtime_df = pd.DataFrame([r.__dict__ for r in runtime_results])

    if stat_df.empty:
        log.warning("No statistics for plotting")
        return

    colors = {
        "dr_greedy": "#D62728", "da": "#1F77B4", "lop": "#2CA02C",
        "go": "#9467BD", "nrs": "#8C564B", "random": "#17BECF",
        "fedsn": "#FF7F0E", "hsfl": "#7F7F7F",
    }

    # Plot 1: Main metrics with 95% CI bars
    metrics = ['latency', 'amse', 'energy']

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))

        metric_df = stat_df[stat_df['metric'] == metric].copy()
        if metric_df.empty:
            continue

        # Sort by mean
        metric_df = metric_df.sort_values('mean')
        x_pos = np.arange(len(metric_df))

        for i, row in metric_df.iterrows():
            algo = row['algorithm']
            ax.bar(
                x_pos[metric_df.index.get_loc(i)],
                row['mean'],
                yerr=[[row['mean'] - row['ci_lower']], [row['ci_upper'] - row['mean']]],
                capsize=5,
                color=colors.get(algo, 'gray'),
                alpha=0.7,
                edgecolor='black',
                linewidth=0.5,
                label=algo
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_df['algorithm'], rotation=45, ha='right')
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} with 95% Bootstrap CI ({len(config.seeds)} seeds)', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        if metric in ['amse']:
            ax.set_yscale('log')

        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"statistics_{metric}_ci.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Plot 2: Runtime comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    runtime_df = runtime_df.sort_values('mean_time')
    x_pos = np.arange(len(runtime_df))

    for i, row in runtime_df.iterrows():
        algo = row['algorithm']
        ax.bar(
            x_pos[runtime_df.index.get_loc(i)],
            row['mean_time'],
            yerr=[[row['mean_time'] - row['ci_lower']], [row['ci_upper'] - row['mean_time']]],
            capsize=5,
            color=colors.get(algo, 'gray'),
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(runtime_df['algorithm'], rotation=45, ha='right')
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Algorithm Runtime with 95% Bootstrap CI', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_statistics.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Box plots for key metrics
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))

        # We need raw data for boxplots - load from CSV
        seed_df = pd.read_csv(os.path.join(config.output_dir, "per_seed_results.csv"))
        if seed_df.empty:
            continue

        # Filter finite
        seed_df = seed_df[np.isfinite(seed_df[metric])]

        if seed_df.empty:
            continue

        algorithms_sorted = sorted(seed_df['algorithm'].unique(),
                                 key=lambda x: seed_df[seed_df['algorithm'] == x][metric].median())

        box_data = [seed_df[seed_df['algorithm'] == a][metric].values for a in algorithms_sorted]

        bp = ax.boxplot(box_data, labels=algorithms_sorted, patch_artist=True)
        for patch, algo in zip(bp['boxes'], algorithms_sorted):
            patch.set_facecolor(colors.get(algo, 'gray'))
            patch.set_alpha(0.6)

        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} Distribution Across Seeds', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        if metric in ['amse']:
            ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"boxplot_{metric}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Plot 4: Pairwise significance tests
    _plot_significance_matrix(stat_results, output_dir)

    log.info(f"Statistics plots saved to {output_dir}")


def _plot_significance_matrix(stat_results: List[StatResult], output_dir: str):
    """Plot pairwise significance test results."""
    stat_df = pd.DataFrame([r.__dict__ for r in stat_results])

    metrics = ['latency', 'amse', 'energy']
    algorithms = sorted(stat_df['algorithm'].unique())

    for metric in metrics:
        metric_df = stat_df[stat_df['metric'] == metric]

        if metric_df.empty:
            continue

        # Create significance matrix
        n = len(algorithms)
        p_matrix = np.ones((n, n))

        # Load raw data for t-tests
        seed_df = pd.read_csv(os.path.join(config.output_dir, "per_seed_results.csv"))

        for i, algo1 in enumerate(algorithms):
            for j, algo2 in enumerate(algorithms):
                if i >= j:
                    continue

                vals1 = seed_df[seed_df['algorithm'] == algo1][metric].values
                vals2 = seed_df[seed_df['algorithm'] == algo2][metric].values

                vals1 = vals1[np.isfinite(vals1)]
                vals2 = vals2[np.isfinite(vals2)]

                if len(vals1) > 1 and len(vals2) > 1:
                    # Welch's t-test (unequal variance)
                    _, p = stats.ttest_ind(vals1, vals2, equal_var=False)
                    p_matrix[i, j] = p
                    p_matrix[j, i] = p

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(p_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.05)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.set_yticklabels(algorithms)
        ax.set_title(f'Pairwise p-values (Welch\'s t-test): {metric.upper()}', fontsize=14)

        # Add text annotations
        for i in range(n):
            for j in range(n):
                if i != j:
                    text = ax.text(j, i, f'{p_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=8)

        plt.colorbar(im, ax=ax, label='p-value')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"significance_{metric}.png"), dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = StatisticsConfig(
        seeds=[42, 123, 456, 789, 999, 111, 222, 333, 444, 555],
        num_bootstrap=10000,
        confidence_level=0.95,
    )

    stat_results, runtime_results = run_statistical_experiments(config=config, budget=20)
    plot_statistics(stat_results, runtime_results)