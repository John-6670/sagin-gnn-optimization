from enum import Enum
import numpy as np
from skyfield.api import Topos, load, EarthSatellite
from typing import List, Optional, Tuple

from simulation.network.channel_model import SaginChannelModel
from simulation.topology.constellation import WalkerConstellation

TS = load.timescale()


class NodeType(Enum):
    SATELLITE = "sat"
    UAV       = "uav"
    GROUND    = "ground"
    CLIENT    = "client"
    

class GroundBSType(Enum):
    URBAN_MACRO = "urban_macro"   # ISD 200 m, height 25 m, gain 17 dBi
    URBAN_MICRO = "urban_micro"   # ISD  50 m, height 10 m, gain  5 dBi
    RURAL       = "rural"         # ISD 1732 m, height 35 m, gain 17 dBi


class ClientMobilityType(Enum):
    STATIONARY = "stationary"   # fixed location         – 0 m/s
    PEDESTRIAN = "pedestrian"   # correlated random walk – 1 m/s
    VEHICULAR  = "vehicular"    # road-constrained       – 15 m/s


# Ground BS physical parameters
BS_CFG: dict = {
    GroundBSType.URBAN_MACRO: dict(
        isd_m=200, height_km=0.025, antenna_gain_dbi=17, power_w=40.0
    ),
    GroundBSType.URBAN_MICRO: dict(
        isd_m=50,  height_km=0.010, antenna_gain_dbi=5,  power_w=10.0
    ),
    GroundBSType.RURAL: dict(
        isd_m=1732, height_km=0.035, antenna_gain_dbi=17, power_w=40.0
    ),
}

# Client mobility speeds
MOB_CFG: dict = {
    ClientMobilityType.STATIONARY: dict(speed_mps=0.0),
    ClientMobilityType.PEDESTRIAN: dict(speed_mps=1.0),    # random walk
    ClientMobilityType.VEHICULAR:  dict(speed_mps=15.0),   # road-aligned
}

# HAP station-keeping constraint (±2 km from home)
HAP_BOX_KM: float = 2.0

# Coverage radii (informational; upper layers may use these for link selection)
SAT_COVERAGE_KM: float  = 1100.0   # 10° min-elevation mask @ 550 km altitude
HAP_COVERAGE_KM: float  = 100.0    # 20 km altitude

# Thermal noise: kTB at 290 K, 20 MHz ≈ 8×10⁻¹⁴ W → round to 1×10⁻¹³ W
THERMAL_NOISE_W: float  = 1e-13


class Node:
    channel_model = SaginChannelModel()
    
    def __init__(self, node_id, node_type: NodeType, position, power=1.0, noise_variance=1e-9, gradient_dim=100,
                tle=None, bs_type: Optional[GroundBSType] = None, mobility_type:  Optional[ClientMobilityType]  = None,
                hap_home: Optional[List[float]] = None,  # [lat°, lon°, alt_km]
        ):
        self.id = node_id
        self.type = node_type
        self.power = power
        self.noise_variance = noise_variance
        self.gradient_dim = gradient_dim
        self.bs_type = bs_type
        self.mobility_type  = (
            mobility_type
            if mobility_type is not None
            else (ClientMobilityType.STATIONARY if node_type == NodeType.CLIENT else None)
        )
        self.antenna_gain_dbi: float = self._default_antenna_gain()
        self.load = 0.0
        
        # Orbital handling
        self.tle = tle
        self.position = np.array(position, dtype=float)
        self.sat_obj  = (
            EarthSatellite(tle[0], tle[1], str(node_id), TS) if tle else None
        )

        # HAP station-keeping state
        self._hap_home      = np.array(hap_home if hap_home is not None else position, dtype=float)
        self._hap_offset_km = np.zeros(3)    # XYZ offset from home [km]
        self._hap_last_t_s: Optional[float] = None

        # Client mobility state
        self._mob_last_t_s: Optional[float] = None
        self._velocity_2d = self._init_client_velocity()  # [deg/s lat, deg/s lon]

        self.channels:  dict = {}
        self.latencies: dict = {}
        
    def _default_antenna_gain(self) -> float:
        if self.bs_type is not None:
            return BS_CFG[self.bs_type]["antenna_gain_dbi"]
        if self.type == NodeType.SATELLITE:
            return 30.0
        if self.type == NodeType.UAV:
            return 10.0
        return 0.0

    def _init_client_velocity(self) -> np.ndarray:
        if self.type != NodeType.CLIENT:
            return np.zeros(2)

        mob = self.mobility_type

        if mob == ClientMobilityType.PEDESTRIAN:
            spd = MOB_CFG[mob]["speed_mps"] / 111_000.0   # m/s → deg/s
            theta = np.random.uniform(0.0, 2.0 * np.pi)
            return np.array([np.cos(theta), np.sin(theta)]) * spd

        if mob == ClientMobilityType.VEHICULAR:
            spd   = MOB_CFG[mob]["speed_mps"] / 111_000.0
            theta = np.random.choice([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
            return np.array([np.cos(theta), np.sin(theta)]) * spd

        return np.zeros(2)

    @staticmethod
    def _skyfield_t_to_seconds(t_now) -> float:
        return 0.0 if t_now is None else float(getattr(t_now, "tt", 0.0) * 86400.0)

    def _topos_xyz(self, lat: float, lon: float, alt_km: float, t_now) -> np.ndarray:
        return (
            Topos(latitude_degrees=lat, longitude_degrees=lon, elevation_m=alt_km * 1000.0)
            .at(t_now).position.km
        )

    def get_position_at(self, t_now=None) -> np.ndarray:
        if t_now is None:
            t_now = TS.now()

        # ── Satellites – predictable, high-speed
        if self.sat_obj is not None:
            return self.sat_obj.at(t_now).position.km   # ~7.586 km/s at 550 km

        now_s = self._skyfield_t_to_seconds(t_now)

        # ── HAPs – bounded station-keeping, slow random drift
        if self.type == NodeType.UAV:
            dt_s = 0.0
            if self._hap_last_t_s is not None:
                dt_s = max(0.0, now_s - self._hap_last_t_s)
            self._hap_last_t_s = now_s

            if dt_s > 0.0:
                # Small Gaussian step (σ ≈ 0.005 km per simulation minute)
                sigma_km = 0.005 * (dt_s / 60.0)
                step     = np.random.normal(0.0, sigma_km, size=3)
                # Hard clamp: enforce ±HAP_BOX_KM station-keeping box
                self._hap_offset_km = np.clip(
                    self._hap_offset_km + step, -HAP_BOX_KM, HAP_BOX_KM
                )

            home_xyz = self._topos_xyz(*self._hap_home, t_now)
            return home_xyz + self._hap_offset_km

        # ── Clients – apply mobility model
        lat, lon, alt_km = self.position

        if self.type == NodeType.CLIENT and self.mobility_type != ClientMobilityType.STATIONARY:
            dt_s = 0.0
            if self._mob_last_t_s is not None:
                dt_s = max(0.0, now_s - self._mob_last_t_s)
            self._mob_last_t_s = now_s

            if dt_s > 0.0:
                if self.mobility_type == ClientMobilityType.PEDESTRIAN:
                    # Correlated random walk: 10 % per-step direction change
                    if np.random.random() < 0.10:
                        spd   = MOB_CFG[ClientMobilityType.PEDESTRIAN]["speed_mps"] / 111_000.0
                        theta = np.random.uniform(0.0, 2.0 * np.pi)
                        self._velocity_2d = np.array([np.cos(theta), np.sin(theta)]) * spd

                elif self.mobility_type == ClientMobilityType.VEHICULAR:
                    # Road-constrained: axis-aligned turns with 5 % probability
                    if np.random.random() < 0.05:
                        spd   = MOB_CFG[ClientMobilityType.VEHICULAR]["speed_mps"] / 111_000.0
                        theta = np.random.choice([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
                        self._velocity_2d = np.array([np.cos(theta), np.sin(theta)]) * spd

                # Advance lat/lon (correct for longitude convergence at high latitudes)
                dlat = self._velocity_2d[0] * dt_s
                dlon = self._velocity_2d[1] * dt_s / max(np.cos(np.radians(lat)), 0.01)
                self.position[0] += dlat
                self.position[1] += dlon

        return self._topos_xyz(self.position[0], self.position[1], self.position[2], t_now)

    def get_velocity_at(self, t_now=None) -> np.ndarray:
        if t_now is None:
            t_now = TS.now()

        if self.sat_obj is not None:
            return self.sat_obj.at(t_now).velocity.km_per_s   # 3-vector

        if self.type == NodeType.UAV:
            return np.zeros(3)   # drift velocity is negligible at HAP scale

        if self.type == NodeType.CLIENT and self.mobility_type != ClientMobilityType.STATIONARY:
            # Convert deg/s → km/s (111 km per degree)
            lat = self.position[0]
            v_lat_km_s = self._velocity_2d[0] * 111.0
            v_lon_km_s = self._velocity_2d[1] * 111.0 * np.cos(np.radians(lat))
            return np.array([v_lat_km_s, v_lon_km_s, 0.0])

        return np.zeros(3)

    def get_channel(self, other, t_now=None):
        return self.channel_model.channel_coefficient(self, other, t_now)

    def compute_snr_to(self, other, t_now=None) -> float:
        return self.channel_model.snr(self, other, t_now)

    def get_latency_to(self, other, t_now=None) -> float:
        """One-way propagation latency [ms]."""
        d_km = np.linalg.norm(
            self.get_position_at(t_now) - other.get_position_at(t_now)
        )
        return d_km * 1_000.0 / 3e8 * 1_000.0


def _hexagonal_hap_positions(n: int, radius_km: float = 60.0) -> List[Tuple[float, float]]:
    """
    Return n (lat°, lon°) positions evenly distributed on a circle of
    `radius_km` around the area centre (0°, 0°).

    This approximates a hexagonal grid spacing; with n=4 the positions form
    a square pattern — a reasonable approximation of the halved 8-HAP hex grid.
    """
    km_per_deg = 111.0
    if n == 1:
        return [(0.0, 0.0)]

    positions = []
    for k in range(n):
        theta = 2.0 * np.pi * k / n
        lat   = radius_km * np.cos(theta) / km_per_deg
        lon   = radius_km * np.sin(theta) / km_per_deg
        positions.append((lat, lon))
    return positions


def generate_nodes(
    num_sats:     int,
    num_uavs:     int,
    num_ground:   int,
    num_clients:  int,
    area_size:    float,
    gradient_dim: int,
    t0=None,
    
) -> List[Node]:
    nodes: List[Node] = []
    side_deg = area_size / 111.0
    half_deg = side_deg / 2.0

    # ── Spatial sampling
    def rand_ll(urban: bool) -> Tuple[float, float]:
        if urban:
            r = 0.35
            lat = np.random.uniform(-half_deg * r, half_deg * r)
            lon = np.random.uniform(-half_deg * r, half_deg * r)
        else:
            lat = np.random.uniform(-half_deg, half_deg)
            lon = np.random.uniform(-half_deg, half_deg)
            # Push outside the urban core (|lat| and |lon| > 35 % of half)
            if abs(lat) < half_deg * 0.35:
                lat = np.sign(lat or 1.0) * np.random.uniform(half_deg * 0.35, half_deg)
        return lat, lon

    # ── 1. Satellites
    # Walker configuration: num_planes × 6 sats/plane ≈ num_sats
    #   e.g. 9 sats → 3 planes × 3 = 9
    tle_list: list = []
    if num_sats > 0:
        n_planes       = max(1, num_sats // 3)
        sats_per_plane = max(1, num_sats // n_planes)
        walker = WalkerConstellation(
            num_planes=n_planes, sats_per_plane=sats_per_plane,
            phasing=1, inclination=53.0, altitude_km=550.0,
        )
        tle_list = walker.generate_tle_list(t0)

    for i in range(num_sats):
        tle      = tle_list[i] if i < len(tle_list) else None
        lat, lon = rand_ll(urban=False)
        nodes.append(Node(
            i, NodeType.SATELLITE, [lat, lon, 550.0],
            power=500.0, noise_variance=THERMAL_NOISE_W,
            gradient_dim=gradient_dim, tle=tle,
        ))

    # ── 2. HAPs (UAVs)
    # Placed in a hexagonal ring at 60 km spacing; altitude fixed at 20 km.
    hap_positions = _hexagonal_hap_positions(num_uavs, radius_km=60.0)
    for j, (lat, lon) in enumerate(hap_positions):
        idx      = num_sats + j
        home_pos = [lat, lon, 20.0]
        nodes.append(Node(
            idx, NodeType.UAV, home_pos[:],
            power=10.0, noise_variance=THERMAL_NOISE_W,
            gradient_dim=gradient_dim, hap_home=home_pos,
        ))

    # ── 3. Ground BSs
    n_macro = round(num_ground * 0.60)
    n_micro = round(num_ground * 0.25)
    n_rural = num_ground - n_macro - n_micro

    bs_plan: List[Tuple[GroundBSType, int, bool]] = [
        (GroundBSType.URBAN_MACRO, n_macro, True),
        (GroundBSType.URBAN_MICRO, n_micro, True),
        (GroundBSType.RURAL,       n_rural, False),
    ]

    bs_idx = num_sats + num_uavs
    for bs_type, count, is_urban in bs_plan:
        cfg = BS_CFG[bs_type]
        for _ in range(count):
            lat, lon = rand_ll(urban=is_urban)
            nodes.append(Node(
                bs_idx, NodeType.GROUND,
                [lat, lon, cfg["height_km"]],
                power=cfg["power_w"], noise_variance=THERMAL_NOISE_W,
                gradient_dim=gradient_dim, bs_type=bs_type,
            ))
            bs_idx += 1

    # ── 4. Clients
    n_stat = round(num_clients * 0.60)
    n_ped  = round(num_clients * 0.20)
    n_veh  = num_clients - n_stat - n_ped

    def client_specs(mob_type: ClientMobilityType, count: int):
        """Build list of (mobility_type, is_urban) tuples."""
        n_urban = round(count * 0.70)
        return (
            [(mob_type, True)]  * n_urban +
            [(mob_type, False)] * (count - n_urban)
        )

    all_specs = (
        client_specs(ClientMobilityType.STATIONARY, n_stat) +
        client_specs(ClientMobilityType.PEDESTRIAN,  n_ped)  +
        client_specs(ClientMobilityType.VEHICULAR,   n_veh)
    )
    np.random.shuffle(all_specs)

    base_idx = num_sats + num_uavs + num_ground
    for k, (mob_type, is_urban) in enumerate(all_specs):
        lat, lon = rand_ll(urban=is_urban)
        power = 0.05 + np.random.uniform(0.0, 0.1)
        nodes.append(Node(
            base_idx + k, NodeType.CLIENT, [lat, lon, 0.001],
            power=power, noise_variance=THERMAL_NOISE_W,
            gradient_dim=gradient_dim, mobility_type=mob_type,
        ))
        nodes[base_idx + k].load = np.random.uniform(0.5, 2.5)

    return nodes