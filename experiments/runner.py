"""
Experiment Runner (Requirements 2-5, 7).

Orchestrates all experiments:
- Pareto Fronts (Req 2)
- MILP Optimality Gap (Req 3)
- Extended Sensitivity (Req 4)
- Statistics Rigor (Req 5)
- FL Integration at end of simulation (Req 7)
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import time
from typing import List, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.topology.nodes import Node, NodeType, generate_nodes
from simulation.config_loader import load_config
from optimization.baselines import dr_selection
from optimization.placement import run_inner_ota_loop
from fl.trainer import run_fl_experiment
from fl.tasks import get_task_registry
from skyfield.api import load

# Import experiment modules
from experiments.pareto_fronts import (
    run_pareto_experiments, plot_pareto_fronts_3d, ParetoConfig
)
from experiments.milp_optimality import (
    run_milp_optimality_experiments, plot_optimality_gap, MILPConfig
)
from experiments.extended_sensitivity import (
    run_extended_sensitivity, plot_sensitivity_results, SensitivityConfig
)
from experiments.statistics_rigor import (
    run_statistical_experiments, plot_statistics, StatisticsConfig
)

log = logging.getLogger(__name__)


def run_fl_at_simulation_end(
    task_name: str = "reddit_nwp",
    num_rounds: int = 20,
    budget: int = 20,
    seed: int = 42,
    csv_dir: str = "results/fl",
):
    """
    Run FL at the end of simulation using DR-Greedy selected servers.

    This implements Requirement 7: FL integration at end of simulation
    when final topology is constant and DR-Greedy selected servers.
    """
    log.info(f"=== Running FL at end of simulation: {task_name}, {num_rounds} rounds ===")

    base_config = load_config("configs/default.yaml")
    sim_config = base_config["simulation"]

    num_sats = sim_config.get("num_sats", 1)
    num_uavs = sim_config.get("num_uavs", 2)
    num_ground = sim_config.get("num_ground", 4)
    num_clients = sim_config.get("num_clients", 20)
    area_size = sim_config.get("area_size", 2000)
    gradient_dim = sim_config.get("gradient_dim", 100)

    ts = load.timescale()
    t_now = ts.now()

    # Generate final topology
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

    from optimization.objective import build_costs
    cost = build_costs(candidates)

    # DR-Greedy server selection
    thresh = base_config.get("algorithm", {}).get("snr_threshold", 0.0)
    alpha = base_config.get("algorithm", {}).get("alpha", 0.5)
    beta = base_config.get("algorithm", {}).get("beta", 0.5)
    delta_list = base_config.get("algorithm", {}).get("delta_list", [0.1, 0.2])
    num_scenarios = base_config.get("simulation", {}).get("num_scenarios", 16)

    log.info("Running DR-Greedy server selection...")
    selected_servers = dr_selection(
        candidates=candidates, clients=clients, budget=budget, cost=cost,
        thresh=thresh, alpha=alpha, beta=beta, delta_list=delta_list,
        N=num_scenarios, t_now=t_now, epsilon=0.1, kappa=0.3
    )

    log.info(f"DR-Greedy selected {len(selected_servers)} servers:")
    for s in selected_servers:
        log.info(f"  {s.type.name}: {s.name}")

    # Run FL experiment
    task_registry = get_task_registry()
    task = task_registry[task_name]
    client_loaders, test_loader = task.get_data_loaders(num_clients=num_clients)

    delta_list = [0.1, 0.2]

    summary = run_fl_experiment(
        task_name=task_name,
        task=task,
        clients=clients,
        servers=selected_servers,
        client_loaders=client_loaders,
        test_loader=test_loader,
        delta_list=delta_list,
        num_rounds=num_rounds,
        use_hybrid=True,
        t_now=t_now,
        csv_dir=csv_dir,
        algo_name="dr_greedy",
    )

    log.info(f"FL Experiment Summary: {summary}")
    return summary


def run_all_experiments(
    run_pareto: bool = True,
    run_milp: bool = True,
    run_sensitivity: bool = True,
    run_statistics: bool = True,
    run_fl: bool = True,
    fl_task: str = "reddit_nwp",
    fl_rounds: int = 20,
    budget: int = 20,
    seed: int = 42,
):
    """
    Run all experiments in sequence.

    Args:
        run_pareto: Run Pareto front experiments
        run_milp: Run MILP optimality gap experiments
        run_sensitivity: Run extended sensitivity experiments
        run_statistics: Run statistical rigor experiments
        run_fl: Run FL at end of simulation
        fl_task: FL task name
        fl_rounds: Number of FL rounds
        budget: Budget constraint
        seed: Random seed
    """
    start_time = time.time()

    os.makedirs("results/experiments", exist_ok=True)
    os.makedirs("plots/experiments", exist_ok=True)

    if run_pareto:
        log.info("=" * 60)
        log.info("Starting Pareto Front Experiments (Req 2)")
        log.info("=" * 60)
        pareto_start = time.time()

        pareto_config = ParetoConfig(
            omega_l_values=[0.1, 0.3, 0.5, 0.7, 0.9],
            amax_values=[10, 20, 30, 40],
            epsilon_values=[0.05, 0.1, 0.2, 0.5],
            num_seeds=3,
        )

        pareto_results = run_pareto_experiments(config=pareto_config, budget=budget)
        plot_pareto_fronts_3d(pareto_results)

        log.info(f"Pareto experiments completed in {time.time() - pareto_start:.1f}s")

    if run_milp:
        log.info("=" * 60)
        log.info("Starting MILP Optimality Gap Experiments (Req 3)")
        log.info("=" * 60)
        milp_start = time.time()

        milp_config = MILPConfig(
            num_clients_list=[5, 6, 7, 8, 9, 10],
            num_seeds=5,
        )

        milp_results = run_milp_optimality_experiments(config=milp_config, budget=budget)
        plot_optimality_gap(milp_results)

        log.info(f"MILP experiments completed in {time.time() - milp_start:.1f}s")

    if run_sensitivity:
        log.info("=" * 60)
        log.info("Starting Extended Sensitivity Experiments (Req 4)")
        log.info("=" * 60)
        sens_start = time.time()

        sens_config = SensitivityConfig(
            num_clients_list=[10, 20, 30, 40, 50],
            num_satellites_list=[1, 2, 3, 4],
            cvar_alpha_list=[0.90, 0.95, 0.99],
            timing_error_std_list=[0.01, 0.05, 0.1, 0.2],
            num_seeds=3,
        )

        sens_results = run_extended_sensitivity(config=sens_config, budget=budget)
        plot_sensitivity_results(sens_results)

        log.info(f"Sensitivity experiments completed in {time.time() - sens_start:.1f}s")

    if run_statistics:
        log.info("=" * 60)
        log.info("Starting Statistics Rigor Experiments (Req 5)")
        log.info("=" * 60)
        stat_start = time.time()

        stat_config = StatisticsConfig(
            seeds=[42, 123, 456, 789, 999, 111, 222, 333, 444, 555],
            num_bootstrap=10000,
            confidence_level=0.95,
        )

        stat_results, runtime_results = run_statistical_experiments(config=stat_config, budget=budget)
        plot_statistics(stat_results, runtime_results, stat_config)

        log.info(f"Statistics experiments completed in {time.time() - stat_start:.1f}s")

    if run_fl:
        log.info("=" * 60)
        log.info("Starting FL Integration at End of Simulation (Req 7)")
        log.info("=" * 60)
        fl_start = time.time()

        run_fl_at_simulation_end(
            task_name=fl_task,
            num_rounds=fl_rounds,
            budget=budget,
            seed=seed,
        )

        log.info(f"FL experiment completed in {time.time() - fl_start:.1f}s")

    total_time = time.time() - start_time
    log.info("=" * 60)
    log.info(f"ALL EXPERIMENTS COMPLETED IN {total_time:.1f}s ({total_time/60:.1f} min)")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run SAGIN GNN optimization experiments")
    parser.add_argument("--pareto", action="store_true", help="Run Pareto front experiments")
    parser.add_argument("--milp", action="store_true", help="Run MILP optimality gap experiments")
    parser.add_argument("--sensitivity", action="store_true", help="Run extended sensitivity experiments")
    parser.add_argument("--statistics", action="store_true", help="Run statistical rigor experiments")
    parser.add_argument("--fl", action="store_true", help="Run FL integration at end of simulation")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--fl-task", type=str, default="reddit_nwp", help="FL task name")
    parser.add_argument("--fl-rounds", type=int, default=20, help="Number of FL rounds")
    parser.add_argument("--budget", type=int, default=20, help="Budget constraint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # If no specific experiment selected, run all
    if not (args.pareto or args.milp or args.sensitivity or args.statistics or args.fl):
        args.all = True

    run_all = args.all

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    run_all_experiments(
        run_pareto=run_all or args.pareto,
        run_milp=run_all or args.milp,
        run_sensitivity=run_all or args.sensitivity,
        run_statistics=run_all or args.statistics,
        run_fl=run_all or args.fl,
        fl_task=args.fl_task,
        fl_rounds=args.fl_rounds,
        budget=args.budget,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()