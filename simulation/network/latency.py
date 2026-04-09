import numpy as np

SPEED_OF_LIGHT_KM_PER_S = 3e5


def propagation_latency(node1, node2):
    """Compute propagation latency in seconds using kilometer distances."""
    distance_km = float(np.linalg.norm(node1.position - node2.position))
    return distance_km / SPEED_OF_LIGHT_KM_PER_S
