import argparse
from typing import List, Dict
import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
import logging
from skyfield.api import load

from simulation.config_loader import load_config
from simulation.topology.nodes import Node, NodeType, generate_nodes
from simulation.topology.aircomp import compute_amse_n
from simulation.topology.patching import hybrid_patch
from simulation.traffic.traffic_generator import generate_spatiotemporal_traffic
from simulation.environment.weather import TwoStateWeatherMarkov
from simulation.evaluation.metrics import compute_e2e_latency, compute_total_energy, compute_cvar
from optimization.placement import predictive_ota_control
from optimization.baselines import da_selection, lop_selection, go_selection, nrs_selection, random_selection, dr_selection
from fl.tasks import get_task_registry
from fl.trainer import FederatedRound
from fl.convergence import convergence_monitor

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy third-party loggers
for _noisy in ("PIL", "matplotlib", "urllib3", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
log = logging.getLogger("run_simulation")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a SAGIN placement simulation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--algorithm", type=str, default=None, help="Override algorithm (e.g., 'all', 'test', 'greedy')")
    parser.add_argument("--fl", action="store_true", help="Run FL experiments")
    parser.add_argument("--task", type=str, default=None, help="FL task to run (e.g. 'reddit_nwp', 'cifar10', 'iot_anomaly'). Runs all if omitted.")
    parser.add_argument("--results-tag", type=str, default="", help="Optional suffix tag for result CSV filenames")
    parser.add_argument("--sensitivity", action="store_true", help="Run DRO sensitivity analysis and ablation")
    return parser.parse_args()


def summarize_selection(selected_servers: List[Node], t_now=None) -> str:
    lines = [f"Selected servers ({len(selected_servers)}):"]
    for server in selected_servers:
        lines.append(
            f"- {server.id} ({server.type.value}) at {server.position.tolist()}"
        )
    return "\n".join(lines)


def build_costs(servers: List[Node]) -> Dict[Node, float]:
    cost = {}
    for s in servers:
        if s.type == NodeType.SATELLITE:
            cost[s] = 10.0
        elif s.type == NodeType.UAV:
            cost[s] = 5.0
        else:
            cost[s] = 1.0
    return cost


def plot_amse_n(total_amse_n, output_dir="plots", log_scale=False):
    """
    Plots AMSE_n per server.
    - ylim is 2x the max height.
    - Scientific notation for very small (< 0.01) and very large (> 1000) values.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for algo_name, algo_data in total_amse_n.items():
        if not algo_data:
            continue

        servers = [str(s.id) for s in algo_data.keys()]
        raw_values = list(algo_data.values())
        finite_values = [v for v in raw_values if np.isfinite(v) and v > 0]
        if finite_values:
            max_val = max(finite_values)
        else:
            max_val = 1.0

        plot_values = [v if np.isfinite(v) and v > 0 else max_val * 10 for v in raw_values]
        if not finite_values:
            plot_values = [1.0 if not np.isfinite(v) or v <= 0 else v for v in raw_values]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(servers, plot_values, color='skyblue', edgecolor='black', alpha=0.8)

        if log_scale:
            ax.set_yscale('log')
            bottom = min([v for v in plot_values if v > 0]) if any(v > 0 for v in plot_values) else 1e-6
            ax.set_ylim(bottom=bottom * 0.1, top=max(plot_values) * 2)
            ax.set_ylabel('AMSE Value (Log Scale)')
        else:
            ax.set_ylim(0, max(plot_values) * 1.3)
            ax.set_ylabel('AMSE Value')

        # Add labels on top of bars
        for i, bar in enumerate(bars):
            raw_height = raw_values[i]
            height = bar.get_height()
            if not np.isfinite(raw_height) or raw_height <= 0:
                label = "INF"
                text_y = height * 1.01 if log_scale else height + (max(plot_values) * 0.03)
            else:
                text_y = height * 1.05 if log_scale else height + (max(plot_values) * 0.03)
                if raw_height < 0.01 or raw_height > 1000:
                    label = f'{raw_height:.4e}'
                else:
                    label = f'{raw_height:.4f}'

            ax.text(bar.get_x() + bar.get_width()/2., text_y,
                    label, ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', rotation=0)
            
        ax.set_title(f'Algorithm: {algo_name.upper()} - AMSE per Server (n)')
        ax.set_xlabel('Server ID')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, f"{algo_name}_amse_n.png")
        plt.savefig(filepath, dpi=300)
        plt.close(fig)
        print(f"Saved: {filepath}")

def plot_amse_kn_grouped(total_avg, total_min, total_max, output_dir="plots", log_scale=False):
    """
    Plots grouped bars (Min, Avg, Max).
    Added: Logic to handle extreme differences in values.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for algo_name in total_avg.keys():
        servers_objs = list(total_avg[algo_name].keys())
        if not servers_objs: continue
            
        servers = [str(s.id) for s in servers_objs]
        avg_vals = [total_avg[algo_name][s] for s in servers_objs]
        min_vals = [total_min[algo_name][s] for s in servers_objs]
        max_vals = [total_max[algo_name][s] for s in servers_objs]
        
        x = np.arange(len(servers))
        width = 0.25 
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        if log_scale:
            ax.set_yscale('log')

        clipped_min = [v if np.isfinite(v) and v > 0 else max([x for x in min_vals if np.isfinite(x) and x > 0], default=1.0) * 10 for v in min_vals]
        clipped_avg = [v if np.isfinite(v) and v > 0 else max([x for x in avg_vals if np.isfinite(x) and x > 0], default=1.0) * 10 for v in avg_vals]
        clipped_max = [v if np.isfinite(v) and v > 0 else max([x for x in max_vals if np.isfinite(x) and x > 0], default=1.0) * 10 for v in max_vals]

        b1 = ax.bar(x - width, clipped_min, width, label='Min', color='#a1d99b', edgecolor='black')
        b2 = ax.bar(x, clipped_avg, width, label='Avg', color='#4292c6', edgecolor='black')
        b3 = ax.bar(x + width, clipped_max, width, label='Max', color='#ef3b2c', edgecolor='black')
        
        def add_labels(bars):
            # Calculate a small dynamic offset for the text
            max_h = max([b.get_height() for b in bars])
            for bar in bars:
                height = bar.get_height()
                # Use scientific notation for very small values
                label = f'{height:.2e}' if height < 0.001 else f'{height:.3f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        label, ha='center', va='bottom', rotation=90, fontsize=8, color='black')
                
        add_labels(b1)
        add_labels(b2)
        add_labels(b3)
        
        ax.set_title(f'Algorithm: {algo_name.upper()} - AMSE_kn Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(servers)
        ax.legend(loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Increase top margin to make room for vertical labels
        ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, f"{algo_name}_amse_kn_grouped.png")
        plt.savefig(filepath, dpi=300)
        plt.close(fig)
        print(f"Saved: {filepath}")


def compute_ota_metrics(selected_servers, clients, target_snr, delta_list, sigma2, gradient_dim, t_now=None):
    ota_results = predictive_ota_control(selected_servers, clients, t_now, target_snr)
    amse_n = {}
    amse_kn_avg = {}
    amse_kn_min = {}
    amse_kn_max = {}
    amse_hybrid_n = {}

    if not selected_servers:
        return amse_n, amse_kn_avg, amse_kn_min, amse_kn_max, amse_hybrid_n

    cascaded_error = np.prod([1 + d for d in delta_list])
    for n in selected_servers:
        snr_dic = {}
        client_amse_values = []

        for k in clients:
            current_snr = k.compute_snr_to(n, t_now)
            optimized = ota_results.get(n.id, {}).get(k.id)
            if optimized is not None:
                optimized_power = optimized["power"]
                new_snr = optimized_power * (current_snr / k.power)
            else:
                new_snr = current_snr

            snr_dic[k] = new_snr
            if new_snr > 0.0:
                amse_kn_val = (k.noise_variance * k.gradient_dim / new_snr) * cascaded_error
            else:
                amse_kn_val = float("inf")

            client_amse_values.append(amse_kn_val)

        amse_n[n] = compute_amse_n(snr_dic, sigma2, gradient_dim)
        _, hybrid_bound, _ = hybrid_patch(clients, n, {c: {n: snr_dic[c]} for c in clients}, delta_list, num_bits=8)
        amse_hybrid_n[n] = hybrid_bound
        if client_amse_values:
            amse_kn_avg[n] = np.mean(client_amse_values)
            amse_kn_min[n] = min(client_amse_values)
            amse_kn_max[n] = max(client_amse_values)

    return amse_n, amse_kn_avg, amse_kn_min, amse_kn_max, amse_hybrid_n


def plot_comparison_amse_n(total_amse_n, output_dir="plots"):
    """
    Compares all algorithms together for AMSE_n.
    X-axis represents algorithms
    Y-axis is the SUM or MEAN of AMSE across selected servers.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    algo_names = list(total_amse_n.keys())
    # We compare the average AMSE_n across the servers each algo selected
    avg_values = [np.mean(list(total_amse_n[algo].values())) if total_amse_n[algo] else 0 
                    for algo in algo_names]
    
    max_val = max(avg_values) if avg_values else 1.0
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(algo_names, avg_values, color=plt.cm.Paired(np.linspace(0, 1, len(algo_names))), edgecolor='black')
    
    ax.set_ylim(0, max_val * 2)
    ax.set_ylabel('Mean AMSE_n across Selected Servers')
    ax.set_title('Algorithm Comparison: Performance Benchmark (AMSE_n)')
    
    for bar in bars:
        height = bar.get_height()
        label = f'{height:.4e}' if (height < 0.01 or height > 1000) else f'{height:.4f}'
        ax.text(bar.get_x() + bar.get_width()/2., height + (max_val * 0.05),
                label, ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_amse_n.png"), dpi=300)
    plt.close(fig)


def plot_comparison_amse_kn(total_avg, total_min, total_max, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    algos = list(total_avg.keys())
    
    # Helper to clean lists of inf/nan
    def get_clean_means(data_dict):
        means = []
        for a in algos:
            vals = list(data_dict[a].values()) if data_dict[a] else [0]
            # Replace inf with a high penalty or remove them to calculate a real mean
            clean_vals = [v if np.isfinite(v) else 0 for v in vals] 
            # If all were inf, we'll handle that below
            means.append(np.mean(clean_vals))
        return np.array(means)

    final_avg = get_clean_means(total_avg)
    final_min = get_clean_means(total_min)
    final_max = get_clean_means(total_max)
    
    # Logic to handle the "Global Max" for ylim when data contains Inf
    # We find the largest FINITE value to set a reasonable scale
    all_metrics = np.concatenate([final_avg, final_min, final_max])
    finite_metrics = all_metrics[np.isfinite(all_metrics)]
    
    if len(finite_metrics) > 0 and np.max(finite_metrics) > 0:
        global_max = np.max(finite_metrics)
    else:
        global_max = 1.0 # Fallback if everything is Inf or Zero

    x = np.arange(len(algos))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    b1 = ax.bar(x - width, final_min, width, label='Overall Min', color='#a1d99b', edgecolor='black')
    b2 = ax.bar(x, final_avg, width, label='Overall Avg', color='#4292c6', edgecolor='black')
    b3 = ax.bar(x + width, final_max, width, label='Overall Max', color='#ef3b2c', edgecolor='black')
    
    # SAFETY CHECK: Set ylim using a finite number
    ax.set_ylim(0, global_max * 2)
    
    def add_labels(bars, original_data_dict):
        for i, bar in enumerate(bars):
            algo_name = algos[i]
            # Check if the original data for this algo was actually infinite
            # We look at the raw values from the dictionary
            raw_vals = list(original_data_dict[algo_name].values()) if original_data_dict[algo_name] else []
            is_inf = any(not np.isfinite(v) for v in raw_vals)
            
            height = bar.get_height()
            
            if is_inf:
                label = "INF (Failed)"
                # Place label at the top of the visible area
                text_pos = global_max * 1.1 
            else:
                label = f'{height:.2e}' if (height < 0.01 or height > 1000) else f'{height:.3f}'
                text_pos = height + (global_max * 0.05)

            ax.text(bar.get_x() + bar.get_width()/2., text_pos,
                    label, ha='center', va='bottom', rotation=90, 
                    fontsize=8, color='red' if is_inf else 'black')

    add_labels(b1, total_min)
    add_labels(b2, total_avg)
    add_labels(b3, total_max)
    
    ax.set_title('Algorithm Comparison: AMSE_kn (Note: INF values capped for visualization)')
    ax.set_xticks(x)
    ax.set_xticklabels(algos)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_amse_kn_grouped.png"), dpi=300)
    plt.close(fig)


# def run_fl_experiments(algorithms, candidates, clients, budget, cost, thresh, alpha, beta, delta_list, output_dir="plots", task_filter=None):
#     os.makedirs(output_dir, exist_ok=True)
#     all_tasks = get_task_registry()
#     if task_filter:
#         if task_filter not in all_tasks:
#             raise ValueError(f"Unknown task '{task_filter}'. Available: {list(all_tasks.keys())}")
#         tasks = {task_filter: all_tasks[task_filter]}
#     else:
#         tasks = all_tasks
#     log.info("=== FL Experiments START | tasks=%s | algorithms=%s | clients=%d | candidates=%d | budget=%d ===",
#              list(tasks.keys()), list(algorithms.keys()), len(clients), len(candidates), budget)
#     for task_name, task in tasks.items():
#         log.info("--- FL Task: %s | local_epochs=%d | lr=%s | optimizer=%s ---",
#                  task_name, task.local_epochs, task.lr, task.optimizer)
#         log.debug("  Loading data loaders for %d clients (seed=123)...", len(clients))
#         client_loaders, test_loader = task.get_data_loaders(len(clients), seed=123)
#         log.debug("  Data loaders ready.")
#         for alg_name, algo in algorithms.items():
#             log.info("  [%s / %s] Running server selection...", task_name, alg_name)
#             selected = algo(
#                 candidates=candidates, clients=clients, budget=budget, cost=cost, thresh=thresh,
#                 alpha=alpha, beta=beta, delta_list=delta_list
#             )
#             if not selected:
#                 log.warning("  [%s / %s] No servers selected — skipping.", task_name, alg_name)
#                 continue
#             log.info("  [%s / %s] Selected %d server(s): %s",
#                      task_name, alg_name, len(selected),
#                      [(s.id, s.type.value) for s in selected])

#             model = task.get_model()
#             log.debug("  [%s / %s] Model built: %s", task_name, alg_name, type(model).__name__)
#             amse_hist = []
#             loss_hist = []
#             acc_hist = []
#             num_rounds = 12
#             for r in range(num_rounds):
#                 log.info("  [%s / %s] Round %2d/%d — computing SNR map...", task_name, alg_name, r+1, num_rounds)
#                 snr_map = {c: {selected[0]: c.compute_snr_to(selected[0])} for c in clients}
#                 snr_vals = [list(v.values())[0] for v in snr_map.values()]
#                 log.debug("    SNR stats: min=%.3e  max=%.3e  mean=%.3e",
#                           min(snr_vals), max(snr_vals), float(np.mean(snr_vals)))
#                 res = FederatedRound(
#                     r, clients, selected, {}, task, model, client_loaders,
#                     test_loader, snr_map, delta_list, use_hybrid=True
#                 )
#                 amse_hist.append(res['amse'])
#                 loss_hist.append(res['loss'])
#                 acc_hist.append(res['accuracy'])
#                 log.info("    Round %2d result: active=%d  p_t=%.3f  amse=%.6f  loss=%.6f  acc=%.4f",
#                          r+1, res['active_clients'], res['p_t'], res['amse'], res['loss'], res['accuracy'])

#             _, logs = convergence_monitor(amse_hist, loss_hist, sigma2=1.0, rho=0.95, gamma=0.5)
#             log.info("  [%s / %s] Convergence summary — final bound=%.6f  final acc=%.4f  final loss=%.6f",
#                      task_name, alg_name, logs[-1]['theoretical_bound'], acc_hist[-1], loss_hist[-1])
#             x = np.arange(1, len(acc_hist)+1)
#             fig, ax = plt.subplots(figsize=(8,4))
#             ax.plot(x, acc_hist, label='accuracy')
#             ax2 = ax.twinx()
#             ax2.plot(x, [l['theoretical_bound'] for l in logs], color='r', label='bound')
#             ax.set_title(f"FL {task_name} - {alg_name}")
#             ax.set_xlabel('round')
#             ax.set_ylabel('acc')
#             ax2.set_ylabel('bound')
#             plt.legend()
#             fig.legend()
#             fig.tight_layout()
#             plot_path = os.path.join(output_dir, f"fl_{task_name}_{alg_name}.png")
#             fig.savefig(plot_path, dpi=200)
#             plt.close(fig)
#             log.info("  [%s / %s] Plot saved: %s", task_name, alg_name, plot_path)
#     log.info("=== FL Experiments DONE ===")


def run_fl_experiments(algorithms, candidates, clients, budget, cost, thresh, alpha, beta, delta_list, output_dir="plots", task_filter=None):
    os.makedirs(output_dir, exist_ok=True)
    all_tasks = get_task_registry()
    if task_filter:
        tasks = {task_filter: all_tasks[task_filter]}
    else:
        tasks = all_tasks

    for task_name, task in tasks.items():
        log.info(f"--- FL Task: {task_name} ---")
        client_loaders, test_loader = task.get_data_loaders(len(clients), seed=123)

        for alg_name, algo in algorithms.items():
            log.info(f"[{alg_name}] Running server placement...")
            selected = algo(
                candidates=candidates, clients=clients, budget=budget, cost=cost,
                thresh=thresh, alpha=alpha, beta=beta, delta_list=delta_list
            )

            if not selected:
                log.warning(f"[{alg_name}] No servers selected.")
                continue

            log.info(f"[{alg_name}] Selected {len(selected)} servers")

            # === Run full FL training ===
            model = task.get_model()
            amse_hist, loss_hist, acc_hist = [], [], []
            num_rounds = 12

            for r in range(num_rounds):
                snr_map = {c: {selected[0]: c.compute_snr_to(selected[0])} for c in clients}
                res = FederatedRound(
                    r, clients, selected, {}, task, model, client_loaders,
                    test_loader, snr_map, delta_list, use_hybrid=True
                )
                amse_hist.append(res['amse'])
                loss_hist.append(res['loss'])
                acc_hist.append(res['accuracy'])

            final_bound, logs = convergence_monitor(amse_hist, loss_hist, sigma2=1.0, rho=0.95, gamma=0.5)
            
            fl_feedback = {
                'final_loss': logs[-1]['loss'],
                'final_accuracy': acc_hist[-1],
                'mean_amse': np.mean(amse_hist),
                'final_bound': final_bound
            }

            log.info(f"[{alg_name}] FL Result → Acc: {acc_hist[-1]:.4f} | Mean AMSE: {fl_feedback['mean_amse']:.6f} | Bound: {final_bound:.4f}")

            # Plot
            x = np.arange(1, len(acc_hist)+1)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(x, acc_hist, label='Accuracy')
            ax2 = ax.twinx()
            ax2.plot(x, [l['theoretical_bound'] for l in logs], color='r', label='Theoretical Bound')
            ax.set_title(f"FL {task_name} - {alg_name}")
            ax.set_xlabel('Round')
            ax.set_ylabel('Accuracy')
            ax2.set_ylabel('Bound')
            fig.legend()
            fig.tight_layout()
            plot_path = os.path.join(output_dir, f"fl_{task_name}_{alg_name}.png")
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)

    log.info("=== FL Experiments DONE ===")


def _run_single_dro_simulation(selected, clients, nodes, tag, duration_hours, 
                              time_step_seconds, outer_interval_minutes, target_snr, N,
                              candidates=None, budget=None, cost=None, thresh=None,
                              alpha=None, beta=None, delta_list=None):
    """Helper to run one DRO configuration simulation"""
    rows = []
    ts = load.timescale()
    start = ts.now()
    weather = TwoStateWeatherMarkov(dt_seconds=time_step_seconds)
    
    print(start, [s.id for s in selected])

    total_steps = int((duration_hours*3600)//time_step_seconds)
    for step in range(total_steps):
        t_now = start + (step*time_step_seconds)/86400.0
        traffic = generate_spatiotemporal_traffic(clients, t_now)
        for c in clients:
            c.load = float(traffic.get(c.id,0.0))
        
        loss_db = weather.atmospheric_loss_db()
        weather.step()
        for n in nodes:
            if hasattr(n,'channel_model'):
                n.channel_model.set_weather_loss_db(loss_db)
        
        avg_traffic = float(np.mean([c.load for c in clients])) if clients else 0.5
        avg_mobility = float(np.mean([np.linalg.norm(s.get_velocity_at(t_now)) for s in selected])) if selected else 0.0
        ota = predictive_ota_control(selected, clients, t_now, target_snr, traffic=avg_traffic, mobility=avg_mobility)
        
        amse_vals = []
        for s in selected:
            snr_dic = {c: c.compute_snr_to(s, t_now) for c in clients}
            amse_vals.append(compute_amse_n(snr_dic, sigma2=10, d=100))
        
        lat = compute_e2e_latency(clients, selected, t_now)
        energy = compute_total_energy(clients, selected, ota, transmission_time=time_step_seconds, t_now=t_now)
        amse = float(np.mean(amse_vals)) if amse_vals else 0.0
        window_size = 30
        # CVaR computed on past window only (exclude current step to avoid trending up with AMSE)
        losses = [r[2] for r in rows[-window_size:]]
        cvar_step = compute_cvar(losses, alpha=0.05) if losses else amse
        rows.append((step, lat, amse, energy, cvar_step))
        
        # Re-selection every outer interval
        if (step * time_step_seconds) % (outer_interval_minutes * 60) == 0 and step > 0:
            old_selected = selected
            selected = dr_selection(
                candidates=candidates,
                clients=clients,
                budget=budget,
                cost=cost,
                thresh=thresh,
                alpha=alpha,
                beta=beta,
                delta_list=delta_list,
                N=N,
                t_now=t_now
            )
            print(t_now, [s.id for s in selected])
            if len(old_selected) != len(selected):
                log.info(f"Server re-selection at step {step}: {len(old_selected)} → {len(selected)} servers")
    
    csv_path = f"results/{tag}_metrics.csv"
    with open(csv_path,'w') as f:
        f.write('step,latency,amse,energy,cvar5\n')
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")
    print(f"   Saved: {csv_path}")


def run_full_sweep(
    algorithms, nodes, clients, candidates, budget, cost, thresh, alpha, beta, delta_list, duration_hours,
    time_step_seconds, outer_interval_minutes, target_snr, N, results_tag="", sensitivity=False
):
    if os.path.exists("results"):
        log.warning("Results directory already exists. New results will be added but old files may be overwritten if names collide.")
        shutil.rmtree("results")
    os.makedirs("results", exist_ok=True)
    
    ts = load.timescale()
    start = ts.now()
    weather = TwoStateWeatherMarkov(dt_seconds=time_step_seconds)
    
    if sensitivity:
        log.info("Running DRO Sensitivity & Ablation Study...")
            
        dr_algo = algorithms['dr_greedy']
        
        # Sensitivity Analysis
        sensitivity_cases = [
            ("budget_20", 20, N, 0.1, 0.3, alpha), ("budget_30", 30, N, 0.1, 0.3, alpha), ("budget_40", 40, N, 0.1, 0.3, alpha),
            ("N_8", budget, 8, 0.1, 0.3, alpha), ("N_32", budget, 32, 0.1, 0.3, alpha), ("N_64", budget, 64, 0.1, 0.3, alpha),
            ("eps_005", budget, N, 0.05, 0.3, alpha), ("eps_015", budget, N, 0.15, 0.3, alpha),
            ("kappa_015", budget, N, 0.1, 0.15, alpha), ("kappa_030", budget, N, 0.1, 0.30, alpha), ("kappa_060", budget, N, 0.1, 0.60, alpha),
            ("alpha_01", budget, N, 0.1, 0.3, 0.1), ("alpha_05", budget, N, 0.1, 0.3, 0.5), ("alpha_1", budget, N, 1.0, 0.3, 1.0),
        ]

        for tag, b, n_scen, eps, kap, alp in sensitivity_cases:
            print(f"→ DRO Sensitivity: {tag}")
            selected = dr_algo(
                candidates=candidates, clients=clients, budget=b, cost=cost,
                thresh=thresh, alpha=alp, beta=beta, delta_list=delta_list,
                N=n_scen, t_now=start, epsilon=eps, kappa=kap
            )
            _run_single_dro_simulation(
                selected, clients, nodes, f"dr_sensitivity_{tag}",
                duration_hours, time_step_seconds, outer_interval_minutes, target_snr, n_scen,
                candidates=candidates, budget=b, cost=cost, thresh=thresh,
                alpha=alpha, beta=beta, delta_list=delta_list
            )

        # Ablation Study
        ablation_cases = {
            "full":      lambda: dr_algo(candidates, clients, budget, cost, thresh, alpha, beta, delta_list, N=N, epsilon=0.1, kappa=0.3, t_now=start),
            "no_gnn":    lambda: dr_algo(candidates, clients, budget, cost, thresh, alpha, beta, delta_list, N=N, epsilon=0.1, kappa=1.0, t_now=start),
            "no_swap":   lambda: dr_algo(candidates, clients, budget, cost, thresh, alpha, beta, delta_list, N=N, epsilon=0.1, kappa=0.3, t_now=start)[:budget],  # skip local swap
        }

        for name, sel_func in ablation_cases.items():
            print(f"→ DRO Ablation: {name}")
            selected = sel_func()
            _run_single_dro_simulation(
                selected, clients, nodes, f"dr_ablation_{name}",
                duration_hours, time_step_seconds, outer_interval_minutes, target_snr, N,
                candidates=candidates, budget=budget, cost=cost, thresh=thresh,
                alpha=alpha, beta=beta, delta_list=delta_list
            )
    else:
        for name, algo in algorithms.items():
            log.info("Running simulation for algorithm: %s", name)
            rows = []
            t_now = start  # use start time for initial selection
            selected = algo(
                candidates=candidates, clients=clients, budget=budget, cost=cost, thresh=thresh,
                alpha=alpha, beta=beta, delta_list=delta_list, N=N, t_now=t_now
            )
            total_steps = int((duration_hours*3600)//time_step_seconds)
            for step in range(total_steps):
                t_now = start + (step*time_step_seconds)/86400.0
                traffic = generate_spatiotemporal_traffic(clients, t_now)
                for c in clients:
                    c.load = float(traffic.get(c.id,0.0))
                    
                loss_db = weather.atmospheric_loss_db()
                weather.step()
                for n in nodes:
                    if hasattr(n,'channel_model'):
                        n.channel_model.set_weather_loss_db(loss_db)
                    
                avg_traffic = float(np.mean([c.load for c in clients])) if clients else 0.5
                avg_mobility = float(np.mean([np.linalg.norm(s.get_velocity_at(t_now)) for s in selected])) if selected else 0.0
                ota = predictive_ota_control(selected, clients, t_now, target_snr, traffic=avg_traffic, mobility=avg_mobility)
    
                amse_vals = []
                for s in selected:
                    snr_dic = {c: c.compute_snr_to(s, t_now) for c in clients}
                    amse_vals.append(compute_amse_n(snr_dic, sigma2=10, d=100))
                    
                lat = compute_e2e_latency(clients, selected, t_now)
                energy = compute_total_energy(clients, selected, ota, transmission_time=time_step_seconds, t_now=t_now)
                if step % 5 == 0 or step == 0:
                    print(f"Step {step:3d} | Algo: {name if 'name' in locals() else 'DRO'} | "
                        f"Servers: {len(selected)} | Energy: {energy:.4f}")
                
                amse = float(np.mean(amse_vals)) if amse_vals else 0.0
                # Use rolling window of last 30 steps for CVaR (matches outer_interval_minutes)
                window_size = 30
                losses = [r[2] for r in rows[-window_size:]] + [amse]
                cvar_step = compute_cvar(losses, alpha=0.05)
                rows.append((step, lat, amse, energy, cvar_step))
                print('Step {}: latency={:.2f}s, amse={:.6f}, energy={:.2f}J, CVaR5%={:.6f}'.format(step, lat, amse, energy, cvar_step))
                    
                # Debug logging for energy anomalies
                if step > 0 and step % 10 == 0:
                    avg_power = energy / len(clients) if clients else 0
                    log.debug(f"Step {step}: energy={energy:.2f}, avg_power_per_client={avg_power:.4f}, num_servers={len(selected)}, num_clients={len(clients)}")
                    
                if (step*time_step_seconds)%(outer_interval_minutes*60) == 0 and step > 0:
                    old_selected = selected
                    selected = algo(
                        candidates=candidates, clients=clients, budget=budget, cost=cost, thresh=thresh,
                        alpha=alpha, beta=beta, delta_list=delta_list, N=N, t_now=t_now
                    )
                    if len(old_selected) != len(selected):
                        log.info(f"Server re-selection at step {step}: {len(old_selected)} → {len(selected)} servers")
            
            print('--- Algorithm {} choosed following servers at last time step ---'.format(name))
            for s in selected:
                print(f"Server {s.id} ({s.type.value}) at position {s.position.tolist()}")
            
            tag = f"_{results_tag}" if results_tag else ""
            csv_path = f"results/{name}{tag}_metrics.csv"
            with open(csv_path,'w') as f:
                f.write('step,latency,amse,energy,cvar5\n')
                for r in rows:
                    f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")


def main():
    args = parse_args()
    log.info("=== SAGIN Simulation START ===")
    log.info("Args: config=%s  algorithm=%s  fl=%s  seed=%s  budget=%s",
             args.config, args.algorithm, args.fl, args.seed, args.budget)

    config = load_config(args.config)
    log.info("Config loaded from %s", args.config)

    # ====================== Config Parameters ======================
    num_sats = config["simulation"].get("num_sats", 1)
    num_uavs = config["simulation"].get("num_uavs", 2)
    num_ground = config["simulation"].get("num_ground", 4)
    num_clients = config["simulation"].get("num_clients", 20)
    area_size = config["simulation"].get("area_size", 2000)
    gradient_dim = config['simulation'].get('gradient_dim', 100)
    sigma2 = config['simulation'].get('sigma2', 10)

    alg_param = args.algorithm or config['simulation'].get('algorithm', 'all')
    duration_hours = config["simulation"].get("duration_hours", 0.0)
    time_step_seconds = config["simulation"].get("time_step_seconds", 10)
    outer_interval_minutes = config["simulation"].get("outer_interval_minutes", 30)
    num_scenarios = config["simulation"].get("num_scenarios", 16)

    alpha = config["algorithm"].get("alpha", 0.5)
    beta = config["algorithm"].get("beta", 0.5)
    delta_list = config["algorithm"].get("delta_list", [0.1, 0.2])
    target_snr = config["algorithm"].get("target_snr", 10.0)
    budget = args.budget or config["algorithm"].get("budget", 20)
    thresh = config["algorithm"].get("snr_threshold", 0.0)

    # ====================== Algorithms Dictionary ======================
    algorithms = {
        "lop": lop_selection,
        "go": go_selection,
        "nrs": nrs_selection,
        "random": random_selection,
        "da": da_selection,
        "dr_greedy": dr_selection,
    }

    # Decide which algorithms to run
    if alg_param in ['all', 'test']:
        active_algorithms = algorithms
        log.info("Running ALL algorithms (test/all mode)")
    else:
        if alg_param not in algorithms:
            raise ValueError(f"Unknown algorithm '{alg_param}'. Available: {list(algorithms.keys())}")
        active_algorithms = {alg_param: algorithms[alg_param]}
        log.info(f"Running single algorithm: {alg_param}")

    log.info("Simulation params: sats=%d uavs=%d ground=%d clients=%d area=%d duration=%.1fh",
             num_sats, num_uavs, num_ground, num_clients, area_size, duration_hours)

    if args.seed is not None:
        np.random.seed(args.seed)
        log.info("Random seed set to %d", args.seed)

    # ====================== Generate Nodes ======================
    ts = load.timescale()
    start = ts.now()

    log.info("Generating nodes...")
    nodes = generate_nodes(
        num_sats=num_sats,
        num_uavs=num_uavs,
        num_ground=num_ground,
        num_clients=num_clients,
        area_size=area_size,
        gradient_dim=gradient_dim,
        t0=start
    )

    clients = [n for n in nodes if n.type == NodeType.CLIENT]
    candidates = [n for n in nodes if n.type != NodeType.CLIENT]

    cost = build_costs(candidates)

    # ====================== FL Experiments ======================
    if args.fl:
        log.info("--- FL Experiments Mode ---")
        run_fl_experiments(
            active_algorithms, candidates, clients, budget, cost, thresh,
            alpha, beta, delta_list, output_dir="plots", task_filter=args.task
        )

    # ====================== Main Simulation Logic ======================
    if duration_hours > 0:
        # Dynamic simulation with time evolution
        log.info(f"Running FULL DYNAMIC SWEEP (duration = {duration_hours}h)")
        run_full_sweep(
            algorithms=active_algorithms,
            nodes=nodes,
            clients=clients,
            candidates=candidates,
            budget=budget,
            cost=cost,
            thresh=thresh,
            alpha=alpha,
            beta=beta,
            delta_list=delta_list,
            duration_hours=duration_hours,
            time_step_seconds=time_step_seconds,
            outer_interval_minutes=outer_interval_minutes,
            target_snr=target_snr,
            N=num_scenarios,
            results_tag=args.results_tag,
            sensitivity=args.sensitivity,
        )
    else:
        # Static test mode (single placement + metrics)
        log.info("Running STATIC TEST mode (duration=0)")
        total_amse_n = {}
        total_amse_kn_avg = {}
        total_amse_kn_min = {}
        total_amse_kn_max = {}

        for name, algo in active_algorithms.items():
            print(f'\n=== Running {name} ===', flush=True)
            selected_servers = algo(
                candidates=candidates,
                clients=clients,
                budget=budget,
                cost=cost,
                thresh=thresh,
                alpha=alpha,
                beta=beta,
                delta_list=delta_list,
                N=num_scenarios
            )

            print(f'--- Algorithm {name} Summary ---')
            print(f"Area size: {area_size} x {area_size}")
            print(f"Total nodes: {len(nodes)} | Clients: {len(clients)}")
            print(f"Budget: {budget} | Selected: {len(selected_servers)} servers")
            print(summarize_selection(selected_servers))

            # Compute OTA-aware metrics
            amse_n, amse_kn_avg, amse_kn_min, amse_kn_max, _ = compute_ota_metrics(
                selected_servers, clients, target_snr, delta_list, sigma2, gradient_dim, t_now=None
            )

            total_amse_n[name] = amse_n
            total_amse_kn_avg[name] = amse_kn_avg
            total_amse_kn_min[name] = amse_kn_min
            total_amse_kn_max[name] = amse_kn_max

        # Generate comparison plots
        print("\nGenerating comparison plots...")
        plot_amse_n(total_amse_n, output_dir="plots", log_scale=True)
        plot_amse_kn_grouped(total_amse_kn_avg, total_amse_kn_min, total_amse_kn_max, output_dir="plots", log_scale=True)
        plot_comparison_amse_n(total_amse_n)
        plot_comparison_amse_kn(total_amse_kn_avg, total_amse_kn_min, total_amse_kn_max)

    log.info("=== SAGIN Simulation END ===")


if __name__ == "__main__":
    main()
