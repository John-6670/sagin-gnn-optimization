from enum import Enum
from typing import List
import numpy as np


class NodeType(Enum):
    SATELLITE = "sat"
    UAV = "uav"
    GROUND = "ground"
    CLIENT = "client"


class Node:
    def __init__(
        self,
        node_id,
        node_type: NodeType,
        position,
        power=1.0,
        noise_variance=1e-9,
        gradient_dim=100,
    ):
        self.id = node_id
        self.type = node_type
        self.position = np.array(position)

        self.power = power
        self.noise_variance = noise_variance
        self.gradient_dim = gradient_dim

        # Channel-related cache
        self.channels = {}   # {other_node_id: complex h}
        self.latencies = {}  # {other_node_id: latency}

    def distance_to(self, other):
        return np.linalg.norm(self.position - other.position)

    def pathloss(self, other, alpha=2.0, fc=2.4e9):
        c = 3e8
        d = self.distance_to(other) + 1e-6
        return (c / (4 * np.pi * fc * d)) ** alpha

    # -----------------------------
    # Channel generation (Rayleigh)
    # -----------------------------
    def generate_channel(self, other):
        fading = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        rho = self.pathloss(other)
        h = rho * fading
        self.channels[other.id] = h
        return h

    # -----------------------------
    # Get channel (cached or new)
    # -----------------------------
    def get_channel(self, other):
        if other.id not in self.channels:
            return self.generate_channel(other)
        return self.channels[other.id]

    # -----------------------------
    # Compute SNR to another node
    # -----------------------------
    def compute_snr_to(self, other):
        """
        Only valid if self is client (transmitter)
        """
        h = self.get_channel(other)
        return self.power * np.abs(h) ** 2 / self.noise_variance

    # -----------------------------
    # Latency model (simple)
    # -----------------------------
    def compute_latency_to(self, other, speed=3e8):
        d = self.distance_to(other)
        latency = d / speed
        self.latencies[other.id] = latency
        return latency

    def get_latency_to(self, other):
        if other.id not in self.latencies:
            return self.compute_latency_to(other)
        return self.latencies[other.id]


def generate_nodes(
    num_sats=3,
    num_uavs=5,
    num_ground=10,
    num_clients=20,
    area_size=2000,
    gradient_dim=100
) -> List[Node]:

    nodes: List[Node] = []

    # Satellites
    for i in range(num_sats):
        x, y = np.random.uniform(0, area_size, 2)
        z = np.random.uniform(500, 2000)
        nodes.append(Node(f"sat-{i}", NodeType.SATELLITE, (x, y, z), gradient_dim=gradient_dim))

    # UAVs
    for i in range(num_uavs):
        x, y = np.random.uniform(0, area_size, 2)
        z = np.random.uniform(20, 50)
        nodes.append(Node(f"uav-{i}", NodeType.UAV, (x, y, z), gradient_dim=gradient_dim))

    # Ground stations
    for i in range(num_ground):
        x, y = np.random.uniform(0, area_size, 2)
        nodes.append(Node(f"gbs-{i}", NodeType.GROUND, (x, y, 0), gradient_dim=gradient_dim))

    # Clients
    for i in range(num_clients):
        x, y = np.random.uniform(0, area_size, 2)
        nodes.append(Node(
            f"client-{i}",
            NodeType.CLIENT,
            (x, y, 0),
            power=1.0,  # only clients transmit,
            gradient_dim=gradient_dim
        ))

    return nodes
