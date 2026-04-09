import argparse
from typing import List

from simulation.config_loader import load_config
from simulation.network.models import SimplePathLossModel
from simulation.traffic.traffic_generator import generate_traffic
from simulation.topology.nodes import Node, NodeType, generate_nodes
from optimization.placement import greedy_placement


def parse_args():
    parser = argparse.ArgumentParser(description="Run a small-scale SAGIN placement simulation.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to configuration YAML file.")
    parser.add_argument("--budget", type=int, default=None, help="Number of servers to deploy.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for scenario generation.")
    return parser.parse_args()


def summarize_selection(selected_servers: List[Node]) -> str:
    lines = [f"Selected servers ({len(selected_servers)}):"]
    for server in selected_servers:
        lines.append(f"- {server.id} ({server.type.value}) at {server.position.tolist()}")
    return "\n".join(lines)


def main():
    args = parse_args()
    config = load_config(args.config)

    num_sats = config["simulation"].get("num_sats", 3)
    num_uavs = config["simulation"].get("num_uavs", 5)
    num_ground = config["simulation"].get("num_ground", 10)
    num_clients = config["traffic"].get("num_clients", 20)
    area_size = config["simulation"].get("area_size", 2000)

    alpha = config["algorithm"].get("alpha", 0.5)
    beta = config["algorithm"].get("beta", 0.5)
    budget = args.budget or config["algorithm"].get("budget", 5)

    if args.seed is not None:
        import numpy as np

        np.random.seed(args.seed)

    nodes = generate_nodes(
        num_sats=num_sats,
        num_uavs=num_uavs,
        num_ground=num_ground,
        num_clients=num_clients,
        area_size=area_size,
    )

    clients = [n for n in nodes if n.type == NodeType.CLIENT]
    candidates = [n for n in nodes if n.type != NodeType.CLIENT]

    network_model = SimplePathLossModel(
        noise_power=config["network"].get("noise_power", 0.001),
        pathloss_exp=config["network"].get("pathloss_exp", 2.0),
    )

    traffic_profile = generate_traffic(num_clients, mean=config["traffic"].get("demand_mean", 1.0))

    selected_servers = greedy_placement(
        candidates=candidates,
        clients=clients,
        network_model=network_model,
        budget=budget,
        alpha=alpha,
        beta=beta,
    )

    print("=== SAGIN Simulation Summary ===")
    print(f"Area size: {area_size} x {area_size} km")
    print(f"Total nodes: {len(nodes)}")
    print(f"Clients: {len(clients)}, candidates: {len(candidates)}")
    print(f"Traffic sample: {traffic_profile[:5].tolist()} ...")
    print(summarize_selection(selected_servers))


if __name__ == "__main__":
    main()