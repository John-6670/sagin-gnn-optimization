import argparse
from typing import List, Dict
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


def plot_barCharts(main_dict):

    os.makedirs("./results", exist_ok=True)
    for outer_key in main_dict:
        inner_dict = main_dict[outer_key]
        data = []
        
        # Collect the object's id, type, and corresponding value
        for obj, value in inner_dict.items():
            label = obj.id
            data.append( (label, value) )
        
        # Sort by object id numerically, then by type
        sorted_data = sorted(data)
        labels = [item[0] for item in sorted_data]
        values = [item[1] for item in sorted_data]
        
        # Create the bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values)
        
        plt.title(f"Bar Chart for {outer_key}")
        plt.xlabel("Object Type-ID")
        plt.ylabel("Float Value")
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        plt.savefig(f'./results/{outer_key}')
        plt.close()


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

        plot_barCharts(total_amse_n)

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
