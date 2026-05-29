from enum import Enum
import numpy as np
from skyfield.api import Topos, load, EarthSatellite

from simulation.network.channel_model import SaginChannelModel
from simulation.topology.constellation import WalkerConstellation

TS = load.timescale()


class NodeType(Enum):
    SATELLITE = "sat"
    UAV = "uav"
    GROUND = "ground"
    CLIENT = "client"


class Node:
    channel_model = SaginChannelModel()
    
    def __init__(self, node_id, node_type: NodeType, position, power=1.0, noise_variance=1e-9, gradient_dim=100, tle=None):
        self.id = node_id
        self.type = node_type
        self.power = power
        self.noise_variance = noise_variance
        self.gradient_dim = gradient_dim
        
        # Orbital handling
        self.tle = tle
        self.position = np.array(position, dtype=float)
        self._drift_ref_t = None
        self._drift_offset = np.zeros(3)

        self.sat_obj = EarthSatellite(tle[0], tle[1], str(node_id), TS) if tle else None

        self.channels = {}
        self.latencies = {}
        
    def _time_seconds(self, t_now):
        return 0.0 if t_now is None else float(getattr(t_now, "tt", 0.0) * 86400.0)

    def get_position_at(self, t_now):
        if t_now is None:
            t_now = TS.now()
        
        if self.sat_obj is not None:
            geocentric = self.sat_obj.at(t_now)
            return geocentric.position.km
        
        lat, lon, alt_km = self.position
        ground_xyz = Topos(latitude_degrees=lat, longitude_degrees=lon, elevation_m=alt_km * 1000.0).at(t_now).position.km
        
        if self.type == NodeType.UAV:
            now_s = self._time_seconds(t_now)
            if self._drift_ref_t is None:
                self._drift_ref_t = now_s
            dt_h = max(0.0, now_s - self._drift_ref_t) / 3600.0
            sigma = 0.5 * dt_h
            self._drift_offset += np.random.normal(0.0, sigma, size=3)
            self._drift_ref_t = now_s
            return ground_xyz + self._drift_offset
        return ground_xyz

    def get_velocity_at(self, t_now):
        if t_now is None:
            t_now = TS.now()
        
        if self.sat_obj is not None:
            return self.sat_obj.at(t_now).velocity.km_per_s
        
        return np.zeros(3)
    
    def get_channel(self, other, t_now=None):
        return self.channel_model.channel_coefficient(self, other, t_now)

    def compute_snr_to(self, other, t_now=None):
        return self.channel_model.snr(self, other, t_now)

    def get_latency_to(self, other, t_now=None):
        pos_self = self.get_position_at(t_now)
        pos_other = other.get_position_at(t_now)
        dist = np.linalg.norm(pos_self - pos_other) * 1000
        return dist / 3e8


def generate_nodes(num_sats, num_uavs, num_ground, num_clients, area_size, gradient_dim, t0=None):
    nodes = []
    
    tle_list = None
    if num_sats > 0:
        walker = WalkerConstellation(num_planes=6, sats_per_plane=6, phasing=1, inclination=53.0, altitude_km=550.0)
        tle_list = walker.generate_tle_list(t0)

    def rand_lat_lon_alt(alt_km):
        side_deg = area_size / 111.0
        lat = np.random.uniform(-side_deg / 2.0, side_deg / 2.0)
        lon = np.random.uniform(-side_deg / 2.0, side_deg / 2.0)
        return np.array([lat, lon, alt_km])

    # Power levels and noise are sized so that Friis free-space SNR is in a
    # meaningful range at SAGIN distances (10–1000 km).
    # noise_variance ≈ thermal noise power: kTB at 20 MHz ≈ 8e-14 W → use 1e-13 W.
    THERMAL_NOISE = 1e-13  # W  (kTB at 290 K, 20 MHz)

    # Generate satellites
    for i in range(num_sats):
        tle = tle_list[i] if tle_list is not None and i < len(tle_list) else None
        pos = rand_lat_lon_alt(550.0)
        nodes.append(Node(i, NodeType.SATELLITE, pos,
                          power=500.0, noise_variance=THERMAL_NOISE,
                          gradient_dim=gradient_dim, tle=tle))

    # Generate UAVs
    for i in range(num_sats, num_sats + num_uavs):
        nodes.append(Node(i, NodeType.UAV, rand_lat_lon_alt(20.0),
                          power=10.0, noise_variance=THERMAL_NOISE,
                          gradient_dim=gradient_dim))

    # Generate ground stations
    for i in range(num_sats + num_uavs, num_sats + num_uavs + num_ground):
        nodes.append(Node(i, NodeType.GROUND, rand_lat_lon_alt(0.1),
                          power=10.0, noise_variance=THERMAL_NOISE,
                          gradient_dim=gradient_dim))

    # Generate clients
    for i in range(num_sats + num_uavs + num_ground, num_sats + num_uavs + num_ground + num_clients):
        nodes.append(Node(i, NodeType.CLIENT, rand_lat_lon_alt(0.1),
                          power=0.1, noise_variance=THERMAL_NOISE,
                          gradient_dim=gradient_dim))
    
    return nodes
