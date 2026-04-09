from .topology.nodes import generate_nodes
from optimization.placement import greedy_placement

if __name__ == "__main__":
    # 1. Generate nodes
    nodes = generate_nodes(num_sats=3, num_uavs=5, num_ground=10)
    clients = [n for n in nodes if n.type == 'ground']
    candidates = [n for n in nodes if n.type != 'ground']

    # 2. Greedy placement
    selected_servers = greedy_placement(candidates, clients, budget=3)

    # 3. Print results
    print("Selected servers:")
    for s in selected_servers:
        print(f"{s.id} ({s.type}) at {s.position}")