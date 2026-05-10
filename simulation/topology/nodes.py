from enum import Enum
from typing import List, Dict
import numpy as np
from skyfield.api import Topos, load, EarthSatellite

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
        position, # For ground/clients: (lat, lon, alt). For Sats: TLE strings.
        power=1.0,
        noise_variance=1e-9,
        gradient_dim=100,
        tle=None
    ):
        self.id = node_id
        self.type = node_type
        self.power = power
        self.noise_variance = noise_variance
        self.gradient_dim = gradient_dim
        
        # Orbital handling
        self.tle = tle
        if self.tle:
            self.sat_obj = EarthSatellite(tle[0], tle[1], node_id, load.timescale())
        else:
            self.position = np.array(position) # [lat, lon, alt]

        self.channels = {}
        self.latencies = {}

    def get_position_at(self, t_now):
        """Returns 3D Cartesian coordinates (ITRS) at time t."""
        if self.type == NodeType.SATELLITE and hasattr(self, 'sat_obj') and self.sat_obj is not None:
            geocentric = self.sat_obj.at(t_now)
            return geocentric.position.km
        return self.position # Simplified for ground nodes or static satellites

    def compute_doppler_shift(self, other, t_now=None, fc=2.4e9):
        """Calculates Doppler shift based on relative velocity"""
        if t_now is None or self.type != NodeType.SATELLITE or not hasattr(self, 'sat_obj') or self.sat_obj is None:
            return 0.0
        
        c = 3e8
        v_rel = np.linalg.norm(self.sat_obj.at(t_now).velocity.km_per_s)
        # Using the formula: fD = (v_rel / c) * fc * cos(theta)
        return (v_rel * 1000 / c) * fc 

    def get_channel(self, other, t_now=None):
        """Generates time-varying channel with Doppler"""
        pos_self = self.get_position_at(t_now)
        pos_other = other.get_position_at(t_now)
        dist = np.linalg.norm(pos_self - pos_other)

        # Pathloss with Doppler shift integration
        f_d = self.compute_doppler_shift(other, t_now)
        fading = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)

        # Phase shift due to Doppler: e^{j2π fD t}
        phase_shift = np.exp(1j * 2 * np.pi * f_d)

        # Use amplitude gain for the channel coefficient.
        # The power gain will be |h|^2 = (3e8 / (4π f d))^2 * |fading|^2.
        amplitude_gain = 3e8 / (4 * np.pi * 2.4e9 * (dist * 1000))

        h = amplitude_gain * fading * phase_shift
        return h

    def compute_snr_to(self, other, t_now=None):
        h = self.get_channel(other, t_now)
        return self.power * np.abs(h) ** 2 / self.noise_variance

    def get_latency_to(self, other, t_now=None):
        """Calculate latency (propagation delay) to another node in seconds"""
        pos_self = self.get_position_at(t_now)
        pos_other = other.get_position_at(t_now)
        dist = np.linalg.norm(pos_self - pos_other) * 1000  # convert km to meters
        c = 3e8  # speed of light in m/s
        return dist / c


def generate_nodes(num_sats, num_uavs, num_ground, num_clients, area_size, gradient_dim):
    """
    Generate a list of nodes for the simulation.
    - Satellites: Use static positions (no TLE for simplicity)
    - UAVs, Ground, Clients: Random positions within area_size
    """
    nodes = []
    
    # Generate satellites
    for i in range(num_sats):
        pos = np.random.uniform(-area_size/2, area_size/2, 3)
        node = Node(i, NodeType.SATELLITE, pos, gradient_dim=gradient_dim)
        nodes.append(node)
    
    # Generate UAVs
    for i in range(num_sats, num_sats + num_uavs):
        pos = np.random.uniform(-area_size/2, area_size/2, 3)
        node = Node(i, NodeType.UAV, pos, gradient_dim=gradient_dim)
        nodes.append(node)
    
    # Generate ground stations
    for i in range(num_sats + num_uavs, num_sats + num_uavs + num_ground):
        pos = np.random.uniform(-area_size/2, area_size/2, 3)
        node = Node(i, NodeType.GROUND, pos, gradient_dim=gradient_dim)
        nodes.append(node)
    
    # Generate clients
    for i in range(num_sats + num_uavs + num_ground, num_sats + num_uavs + num_ground + num_clients):
        pos = np.random.uniform(-area_size/2, area_size/2, 3)
        node = Node(i, NodeType.CLIENT, pos, gradient_dim=gradient_dim)
        nodes.append(node)
    
    return nodes