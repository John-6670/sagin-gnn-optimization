import argparse
from typing import List, Dict
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import os

from simulation.config_loader import load_config
from simulation.topology.nodes import Node, NodeType, generate_nodes
from simulation.topology.aircomp import compute_amse_kn, compute_amse_n
from optimization.placement import greedy_server_selection
from optimization.baselines import lop_selection, go_selection, nrs_selection, random_selection


def parse_args():
    parser = argparse.ArgumentParser(description="Run a SAGIN placement simulation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def summarize_selection(selected_servers: List[Node]) -> str:
    lines = [f"Selected servers ({len(selected_servers)}):"]
    for server in selected_servers:
        lines.append(
            f"- {server.id} ({server.type.value}) at {server.position.tolist()}"
        )
    return "\n".join(lines)


# Cost model (simple for now)
def build_costs(servers: List[Node]) -> Dict[Node, float]:
    """
    Simple cost model:
    - satellite > uav > ground
    """
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
            
        # Extract server IDs and values
        servers = [str(s.id) for s in algo_data.keys()]
        values = list(algo_data.values())
        max_val = max(values) if values else 1.0
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(servers, values, color='skyblue', edgecolor='black', alpha=0.8)
        
        # Set Y-axis scale and limits
        if log_scale:
            ax.set_yscale('log')
            # In log scale, we jump an order of magnitude to provide 
            # visual space similar to the "2x" request.
            ax.set_ylim(bottom=min(values)*0.1 if min(values) > 0 else None, 
                        top=max_val * 2) 
            ax.set_ylabel('AMSE Value (Log Scale)')
        else:
            ax.set_ylim(0, max_val * 1.3)  # Strictly 2x the max height
            ax.set_ylabel('AMSE Value')

        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            
            # Position text slightly above the bar
            # 2% of max_val is a good padding for linear scales
            text_y = height * 1.05 if log_scale else height + (max_val * 0.03)
            
            # Formatting logic: 
            # Use scientific notation if too small or too large
            if height < 0.01 or height > 1000:
                label = f'{height:.4e}'
            else:
                label = f'{height:.4f}'
            
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

        b1 = ax.bar(x - width, min_vals, width, label='Min', color='#a1d99b', edgecolor='black')
        b2 = ax.bar(x, avg_vals, width, label='Avg', color='#4292c6', edgecolor='black')
        b3 = ax.bar(x + width, max_vals, width, label='Max', color='#ef3b2c', edgecolor='black')
        
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


def main():
    args = parse_args()
    config = load_config(args.config)

    num_sats = config["simulation"].get("num_sats", 1)
    num_uavs = config["simulation"].get("num_uavs", 2)
    num_ground = config["simulation"].get("num_ground", 4)
    num_clients = config["simulation"].get("num_clients", 20)
    area_size = config["simulation"].get("area_size", 2000)
    gradient_dim = config['simulation'].get('gradient_dim', 100)
    sigma2 = config['simulation'].get('sigma2', 10)
    alg = config['simulation'].get('algorithm', 'all')

    alpha = config["algorithm"].get("alpha", 0.5)
    beta = config["algorithm"].get("beta", 0.5)
    delta_list = config["algorithm"].get("delta_list", [0.1, 0.2])

    budget = args.budget or config["algorithm"].get("budget", 20)

    thresh = config["algorithm"].get("snr_threshold", 0.0)
    

    algorithms = {
        "greedy": greedy_server_selection,
        "lop": lop_selection,
        "go": go_selection,
        "nrs": nrs_selection,
        "random": random_selection,
    }

    if args.seed is not None:
        np.random.seed(args.seed)

    nodes = generate_nodes(
        num_sats=num_sats,
        num_uavs=num_uavs,
        num_ground=num_ground,
        num_clients=num_clients,
        area_size=area_size,
        gradient_dim=gradient_dim
    )

    clients = [n for n in nodes if n.type == NodeType.CLIENT]
    candidates = [n for n in nodes if n.type != NodeType.CLIENT]
    n_can = len(candidates)
    g_can = len([n for n in candidates if n.type == NodeType.GROUND])

    cost = build_costs(candidates)

    if alg in ['all', 'test']:
        total_amse_n = {}
        total_amse_kn_avg = {}
        total_amse_kn_max = {}
        total_amse_kn_min = {}
        for name, algo in algorithms.items():
            print(f'begin algorithm {name}')
            selected_servers = algo(
                candidates=candidates,
                clients=clients,
                budget=budget,
                cost=cost,
                thresh=thresh,
                alpha=alpha,
                beta=beta,
                delta_list=delta_list
            )
            print(f'--- Algorithm {name} Summary ---')
            print(f"Area size: {area_size} x {area_size}")
            print(f"Total nodes: {len(nodes)}")
            print(f"Clients: {len(clients)}, candidates: {n_can if name != 'go' else g_can}")
            print(f"Alpha={alpha}, Beta={beta}")
            print(f"Budget: {budget}")
            print(summarize_selection(selected_servers))

            if alg == 'test':
                amse_n = {}
                for n in selected_servers:
                    snr_dic = {k: k.compute_snr_to(n) for k in clients}
                    amse_n[n] = compute_amse_n(snr_dic, sigma2, gradient_dim)
                total_amse_n[name] = amse_n

                amse_kn = defaultdict(dict)
                amse_kn_avg = {}
                amse_kn_min = {}
                amse_kn_max = {}
                for n in selected_servers:
                    min = None
                    max = None
                    sum = 0
                    for k in clients:
                        amse_kn[n][k] = compute_amse_kn(k, n, delta_list)
                        sum += amse_kn[n][k]
                        if min is None or amse_kn[n][k] < min:
                            min = amse_kn[n][k]
                        if max is None or amse_kn[n][k] > max:
                            max = amse_kn[n][k]
                    amse_kn_avg[n] = sum/len(clients)
                    amse_kn_max[n] = max
                    amse_kn_min[n] = min
                
                total_amse_kn_avg[name] = amse_kn_avg
                total_amse_kn_min[name] = amse_kn_min
                total_amse_kn_max[name] = amse_kn_max

        if alg == 'test':
            print("\nGenerating and saving plots...")
            # 1. Plot single AMSE_n per server
            plot_amse_n(total_amse_n, output_dir="plots", log_scale=True)
            
            # 2. Plot grouped Min/Avg/Max AMSE_kn per server
            plot_amse_kn_grouped(total_amse_kn_avg, total_amse_kn_min, total_amse_kn_max, output_dir="plots", log_scale=True)

    else:
        algo = algorithms[alg]
        print(f'begin algorithm {alg}')
        selected_servers = algo(
            candidates=candidates,
            clients=clients,
            budget=budget,
            cost=cost,
            thresh=thresh,
            alpha=alpha,
            beta=beta,
            delta_list=delta_list
        )
        print(f'--- Algorithm {alg} Summary ---')
        print(f"Area size: {area_size} x {area_size}")
        print(f"Total nodes: {len(nodes)}")
        print(f"Clients: {len(clients)}, candidates: {n_can if alg != 'go' else g_can}")
        print(f"Alpha={alpha}, Beta={beta}")
        print(f"Budget: {budget}")
        print(summarize_selection(selected_servers))


if __name__ == "__main__":
    main()
