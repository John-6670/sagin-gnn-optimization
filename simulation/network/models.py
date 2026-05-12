from abc import ABC, abstractmethod
from simulation.network.channel_model import SaginChannelModel


class NetworkModel(ABC):
    @abstractmethod
    def snr(self, node1, node2, t_now=None):
        raise NotImplementedError

    @abstractmethod
    def latency(self, node1, node2, t_now=None):
        raise NotImplementedError


class PathLossModel(NetworkModel):
    """Backward-compatible wrapper now delegating to the realistic SAGIN model."""

    def __init__(self):
        self.model = SaginChannelModel()

    def snr(self, node1, node2, t_now=None):
        return self.model.snr(node1, node2, t_now)

    def latency(self, node1, node2, t_now=None):
        return node1.get_latency_to(node2, t_now)
