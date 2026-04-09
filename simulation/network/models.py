from abc import ABC, abstractmethod
import numpy as np


class NetworkModel(ABC):
    """Abstract network model for SAGIN link metrics."""

    @abstractmethod
    def snr(self, node1, node2):
        raise NotImplementedError

    @abstractmethod
    def latency(self, node1, node2):
        raise NotImplementedError


class SimplePathLossModel(NetworkModel):
    """Simplified path-loss model for early prototyping."""

    def __init__(self, noise_power: float = 1e-3, pathloss_exp: float = 2.0, tx_power: float = 1.0):
        self.noise_power = noise_power
        self.pathloss_exp = pathloss_exp
        self.tx_power = tx_power

    def _distance(self, node1, node2) -> float:
        return float(np.linalg.norm(node1.position - node2.position))

    def snr(self, node1, node2) -> float:
        d = max(self._distance(node1, node2), 1e-3)
        return self.tx_power * (d**-self.pathloss_exp) / self.noise_power

    def latency(self, node1, node2) -> float:
        d = self._distance(node1, node2)
        return d / 3e5
