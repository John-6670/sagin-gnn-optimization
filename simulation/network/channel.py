from simulation.topology.nodes import NodeType


def link_properties(source, destination):
    """Return nominal link properties based on node types."""
    if source.type == NodeType.SATELLITE or destination.type == NodeType.SATELLITE:
        return {"link_type": "satellite", "bandwidth_mhz": 100, "latency_ms": 30}
    if source.type == NodeType.UAV or destination.type == NodeType.UAV:
        return {"link_type": "aerial", "bandwidth_mhz": 50, "latency_ms": 15}
    return {"link_type": "ground", "bandwidth_mhz": 20, "latency_ms": 5}
