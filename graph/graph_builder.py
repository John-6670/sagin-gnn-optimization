import networkx as nx
from typing import List, Tuple

from simulation.topology.nodes import Node, NodeType, generate_nodes
from simulation.network.models import SimplePathLossModel
from optimization.placement import greedy_placement
from optimization.objective import utility


def build_graph(num_sats=3, num_uavs=5, num_ground=10, num_clients=20, area_size=2000) -> Tuple[nx.Graph, List[Node]]:
    """Build a prototype heterogeneous SAGIN graph for placement analysis."""
    nodes = generate_nodes(
        num_sats=num_sats,
        num_uavs=num_uavs,
        num_ground=num_ground,
        num_clients=num_clients,
        area_size=area_size,
    )

    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.id, node_type=node.type.value, position=node.position.tolist())

    network_model = SimplePathLossModel()
    for i, source in enumerate(nodes):
        for target in nodes[i + 1 :]:
            latency = network_model.latency(source, target)
            graph.add_edge(source.id, target.id, latency=latency)

    return graph, nodes


def main():
    graph, nodes = build_graph()
    clients = [n for n in nodes if n.type == NodeType.CLIENT]
    candidates = [n for n in nodes if n.type != NodeType.CLIENT]

    selected = greedy_placement(candidates, clients, budget=3, utility_function=utility, cost= dict[Node, float],
                                network_model=SimplePathLossModel(), thresh=1e-2, alpha=0.5, beta=0.5)

    print(f"Graph nodes: {len(graph.nodes)}")
    print(f"Selected {len(selected)} candidate servers:")
    for server in selected:
        print(f"- {server.id} ({server.type.value})")


if __name__ == "__main__":
    main()
