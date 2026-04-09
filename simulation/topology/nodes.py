from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import numpy as np


class NodeType(Enum):
    SATELLITE = "sat"
    UAV = "uav"
    GROUND = "ground"
    CLIENT = "client"


@dataclass
class Node:
    id: str
    type: NodeType
    position: np.ndarray


def generate_nodes(
    num_sats: int = 3,
    num_uavs: int = 5,
    num_ground: int = 10,
    num_clients: int = 20,
    area_size: int = 2000,
) -> List[Node]:
    """Generate a simple SAGIN node topology for early-stage simulation."""
    nodes: List[Node] = []

    # Satellites at Low Earth Orbit altitudes
    for i in range(num_sats):
        x, y = np.random.uniform(0, area_size, 2)
        z = np.random.uniform(500, 2000)
        nodes.append(Node(f"sat-{i}", NodeType.SATELLITE, np.array((x, y, z))))

    # UAVs and HAPs in the aerial segment
    for i in range(num_uavs):
        x, y = np.random.uniform(0, area_size, 2)
        z = np.random.uniform(20, 50)
        nodes.append(Node(f"uav-{i}", NodeType.UAV, np.array((x, y, z))))

    # Ground base stations
    for i in range(num_ground):
        x, y = np.random.uniform(0, area_size, 2)
        nodes.append(Node(f"gbs-{i}", NodeType.GROUND, np.array((x, y, 0.0))))

    # Mobile clients on the ground
    for i in range(num_clients):
        x, y = np.random.uniform(0, area_size, 2)
        nodes.append(Node(f"client-{i}", NodeType.CLIENT, np.array((x, y, 0.0))))

    return nodes
