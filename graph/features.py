from typing import List, Dict

from simulation.topology.nodes import NodeType, Node


def extract_node_features(nodes: List[Node]) -> List[Dict[str, float]]:
    """Extract basic heterogeneous node features for early GNN prototypes."""
    features = []
    for node in nodes:
        features.append(
            {
                "is_satellite": float(node.type == NodeType.SATELLITE),
                "is_uav": float(node.type == NodeType.UAV),
                "is_ground": float(node.type == NodeType.GROUND),
                "is_client": float(node.type == NodeType.CLIENT),
                "x_position": float(node.position[0]),
                "y_position": float(node.position[1]),
                "z_position": float(node.position[2]),
            }
        )
    return features
