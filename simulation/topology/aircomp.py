from typing import Sequence

from simulation.network.models import NetworkModel
from simulation.topology.nodes import Node


def compute_amse(
    server: Node,
    clients: Sequence[Node],
    network_model: NetworkModel,
    gradient_dim: int = 10,
) -> float:
    """Compute a prototype AirComp AMSE score for a candidate server."""
    inverse_snr_sum = 0.0
    for client in clients:
        snr_value = max(network_model.snr(server, client), 1e-6)
        inverse_snr_sum += 1.0 / snr_value

    return gradient_dim * inverse_snr_sum
