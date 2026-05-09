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
        if self.type == NodeType.SATELLITE:
            geocentric = self.sat_obj.at(t_now)
            return geocentric.position.km
        return self.position # Simplified for ground nodes

    def compute_doppler_shift(self, other, t_now, fc=2.4e9):
        """Calculates Doppler shift based on relative velocity"""
        c = 3e8
        if self.type != NodeType.SATELLITE: return 0.0
        
        v_rel = np.linalg.norm(self.sat_obj.at(t_now).velocity.km_per_s)
        # Using the formula: fD = (v_rel / c) * fc * cos(theta)
        return (v_rel * 1000 / c) * fc 

    def get_channel(self, other, t_now):
        """Generates time-varying channel with Doppler"""
        pos_self = self.get_position_at(t_now)
        pos_other = other.get_position_at(t_now)
        dist = np.linalg.norm(pos_self - pos_other)
        
        # Pathloss with Doppler shift integration
        f_d = self.compute_doppler_shift(other, t_now)
        fading = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        
        # Phase shift due to Doppler: e^{j2π fD t}
        phase_shift = np.exp(1j * 2 * np.pi * f_d) 
        rho = (3e8 / (4 * np.pi * 2.4e9 * (dist * 1000))) ** 2.0
        
        h = rho * fading * phase_shift
        return h

    def compute_snr_to(self, other, t_now):
        h = self.get_channel(other, t_now)
        return self.power * np.abs(h) ** 2 / self.noise_variance